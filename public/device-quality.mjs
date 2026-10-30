/** Bounded RGBA statistics. These are frame quality/color cues, never mushroom IDs. */
export function qualityMetrics(data,width,height,{sourceWidth=width,sourceHeight=height,processingMs=0}={}) {
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>640||height>640||data.length!==width*height*4)throw new Error('Quality sample must be RGBA, at most 640 pixels per side.');
  const n=width*height,y=new Float64Array(n);let sum=0,sum2=0,chroma=0,rSum=0,gSum=0,bSum=0;
  for(let i=0;i<n;i++){const k=i*4,r=data[k],g=data[k+1],b=data[k+2],v=.2126*r+.7152*g+.0722*b;y[i]=v;sum+=v;sum2+=v*v;chroma+=Math.max(r,g,b)-Math.min(r,g,b);rSum+=r;gSum+=g;bSum+=b;}
  const mr=rSum/n,mg=gSum/n,mb=bSum/n;let outliers=0,lap=0,lap2=0,count=0;
  for(let i=0;i<n;i++){const k=i*4,dr=data[k]-mr,dg=data[k+1]-mg,db=data[k+2]-mb;if(Math.sqrt((dr*dr+dg*dg+db*db)/3)>40)outliers++;}
  for(let row=1;row<height-1;row++)for(let col=1;col<width-1;col++){const i=row*width+col,v=y[i-width]+y[i+width]+y[i-1]+y[i+1]-4*y[i];lap+=v;lap2+=v*v;count++;}
  return {algorithm:'rgba-quality-v1',width:sourceWidth,height:sourceHeight,sampleWidth:width,sampleHeight:height,
    meanLuminance:sum/n,luminanceStd:Math.sqrt(Math.max(0,sum2/n-(sum/n)**2)),laplacianVariance:count?Math.max(0,lap2/count-(lap/count)**2):0,
    meanChroma:chroma/n,colorOutlierFraction:outliers/n,processingMs};
}

export async function analyzeImageBlob(blob) {
  if(blob.size>4*1024*1024)throw new Error('Image exceeds 4 MiB.');
  const start=performance.now(),bitmap=await createImageBitmap(blob,{imageOrientation:'none'});
  try {
    if(bitmap.width*bitmap.height>8_000_000)throw new Error('Image exceeds 8 megapixels.');
    const scale=Math.min(1,640/Math.max(bitmap.width,bitmap.height)),width=Math.max(1,Math.round(bitmap.width*scale)),height=Math.max(1,Math.round(bitmap.height*scale));
    const canvas=typeof OffscreenCanvas!=='undefined'?new OffscreenCanvas(width,height):Object.assign(document.createElement('canvas'),{width,height});
    const ctx=canvas.getContext('2d',{willReadFrequently:true});if(!ctx)throw new Error('Canvas pixel access is unavailable.');
    ctx.drawImage(bitmap,0,0,width,height);const result=qualityMetrics(ctx.getImageData(0,0,width,height).data,width,height,{sourceWidth:bitmap.width,sourceHeight:bitmap.height});result.processingMs=performance.now()-start;return result;
  } finally {bitmap.close();}
}
