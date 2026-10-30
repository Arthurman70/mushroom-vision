const categories=new Set(['immediate_investigation','habitat_followup','unlocated_followup']);
export function confirmedIdentification(find,groupLabel=id=>id){
  if(find?.reviewStatus!=='reviewed'||!find.latestReview)return null;
  return find.latestReview.species||(find.latestReview.groupId?groupLabel(find.latestReview.groupId):null);
}
export function investigationView(find){
  const metadata=find?.metadata??{},stored=metadata.investigation;
  if(!stored||!categories.has(stored.routingCategory))return null;
  const colorOnly=metadata.cueType==='color_salience';
  const category=colorOnly?'habitat_followup':stored.routingCategory;
  const label=category==='immediate_investigation'?'Immediate investigation requested':category==='habitat_followup'?'Habitat follow-up':'Locate before investigation';
  const evidence=stored.rangeEvidence??{};
  const range=evidence.valueM??evidence.rangeM;
  const measured=Number.isFinite(range)&&range>0&&['known','measured'].includes(evidence.status)&&evidence.synthetic!==true;
  const rangeText=measured?`Reported capture range: ${range.toFixed(1)} m.`:'Physical range is unknown. Pixel size is not a measured distance.';
  const photoText=({
    closer_photo_requested:'A closer original photo is requested.',
    waiting_for_capture_ack:'Waiting for the requested original photo acknowledgment.',
    zoom_required:'Optical zoom is needed to meet the close-photo framing.',
    zoom_required_or_manual_closeup:'Optical zoom or a manual close-up is needed to fill the frame.',
    manual_closeup_required:'A person needs to capture the close-up.',
    manual_closeup_camera_pitch_unreachable:'The fixed camera cannot center this target; a manual close-up is needed.',
    capture_ack_missing_manual_closeup_required:'The requested photo was not acknowledged; the close-up is still pending.',
    simulated_safe_photo_manual_closeup_required:'A safe-view photo was simulated; the close-up still needs zoom or a person.',
    safe_photo_acknowledged_manual_closeup_required:'A safe-view photo was acknowledged; a manual close-up is still required.',
    simulated_frame_filling_photo:'Close-photo framing was acknowledged in simulation only. No physical photo was captured.',
    photo_acknowledged_pending_human_confirmation:'A photo was acknowledged; human confirmation remains pending.',
    simulated_capture_requested:'A simulated photo request exists; no real image has been captured.'
  })[stored.photoStatus]||'Close-photo completion has not been verified.';
  return {category,label,colorOnly,rangeText,photoText,reason:colorOnly?'A color difference is a search clue; mushroom presence is unconfirmed.':category==='habitat_followup'?'Keep this as a search lead until a closer view confirms what is present.':'Investigate and collect a close view for human confirmation.',notice:'Recorded investigation request. Movement still requires fresh position, depth and a clear route.'};
}
