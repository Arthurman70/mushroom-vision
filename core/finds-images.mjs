import { createHash } from 'node:crypto';
import { ApiError } from './validation.mjs';

export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 20_000_000;
export const MAX_IMAGE_BODY = 12 * 1024 * 1024;

function dimensions(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 ||
      width > 20000 || height > 20000 || width * height > MAX_IMAGE_PIXELS) {
    throw new ApiError('Images must be at most 20 megapixels and 20000 pixels per side.', 413, 'IMAGE_DIMENSIONS');
  }
  return { width, height };
}

function pngHeader(bytes) {
  if (bytes.length < 45 || bytes.readUInt32BE(8) !== 13 || bytes.toString('ascii', 12, 16) !== 'IHDR') {
    throw new ApiError('PNG header is invalid.', 400, 'INVALID_IMAGE');
  }
  const size = dimensions(bytes.readUInt32BE(16), bytes.readUInt32BE(20));
  if (bytes[26] !== 0 || bytes[27] !== 0 || bytes[28] > 1) throw new ApiError('Unsupported PNG encoding.', 400, 'INVALID_IMAGE');
  let offset = 8, data = false, end = false;
  while (offset + 12 <= bytes.length) {
    const length = bytes.readUInt32BE(offset), type = bytes.toString('ascii', offset + 4, offset + 8);
    if (length > MAX_IMAGE_BYTES || offset + 12 + length > bytes.length) throw new ApiError('PNG is truncated.', 400, 'INVALID_IMAGE');
    if (type === 'acTL') throw new ApiError('Use a still PNG or JPEG image.', 400, 'INVALID_IMAGE');
    if (type === 'IDAT') data = true;
    offset += length + 12;
    if (type === 'IEND') { end = length === 0 && offset === bytes.length; break; }
  }
  if (!data || !end) throw new ApiError('PNG data is incomplete.', 400, 'INVALID_IMAGE');
  return size;
}

function jpegHeader(bytes) {
  if (bytes.length < 8 || bytes.at(-2) !== 0xff || bytes.at(-1) !== 0xd9) throw new ApiError('JPEG is incomplete.', 400, 'INVALID_IMAGE');
  let offset = 2, size = null;
  const sof = new Set([0xc0, 0xc1, 0xc2]);
  while (offset < bytes.length - 2) {
    if (bytes[offset++] !== 0xff) throw new ApiError('JPEG marker is invalid.', 400, 'INVALID_IMAGE');
    while (bytes[offset] === 0xff) offset++;
    const marker = bytes[offset++];
    if (marker === 0xda) {
      if (!size) throw new ApiError('JPEG has no supported frame header.', 400, 'INVALID_IMAGE');
      if (offset + 2 > bytes.length - 2) throw new ApiError('JPEG scan is incomplete.', 400, 'INVALID_IMAGE');
      const scanLength = bytes.readUInt16BE(offset);
      if (scanLength < 6 || offset + scanLength >= bytes.length - 2) throw new ApiError('JPEG scan is incomplete.', 400, 'INVALID_IMAGE');
      return size;
    }
    if ([0xd8, 0xd9, 0x00].includes(marker) || offset + 2 > bytes.length) throw new ApiError('JPEG marker is invalid.', 400, 'INVALID_IMAGE');
    const length = bytes.readUInt16BE(offset);
    if (length < 2 || offset + length > bytes.length) throw new ApiError('JPEG is truncated.', 400, 'INVALID_IMAGE');
    if (sof.has(marker)) {
      if (length < 8 || bytes[offset + 2] !== 8) throw new ApiError('Unsupported JPEG frame.', 400, 'INVALID_IMAGE');
      size = dimensions(bytes.readUInt16BE(offset + 5), bytes.readUInt16BE(offset + 3));
    }
    offset += length;
  }
  throw new ApiError('JPEG scan is missing.', 400, 'INVALID_IMAGE');
}

export function validateImage(input) {
  let encoded = input.imageBase64, declared = input.mimeType;
  if (typeof encoded !== 'string') throw new ApiError('imageBase64 is required.');
  const dataUri = /^data:(image\/(?:jpeg|png));base64,/.exec(encoded);
  if (dataUri) {
    if (declared && declared !== dataUri[1]) throw new ApiError('Image MIME types disagree.');
    declared = dataUri[1]; encoded = encoded.slice(dataUri[0].length);
  }
  if (!encoded.length || encoded.length > Math.ceil(MAX_IMAGE_BYTES / 3) * 4) throw new ApiError('Image exceeds 8 MiB.', 413, 'IMAGE_TOO_LARGE');
  if (encoded.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) throw new ApiError('Image is not valid base64.', 400, 'INVALID_IMAGE');
  const bytes = Buffer.from(encoded, 'base64');
  if (bytes.length > MAX_IMAGE_BYTES || bytes.toString('base64') !== encoded) throw new ApiError('Image is not canonical base64.', 400, 'INVALID_IMAGE');
  let mimeType, extension, size;
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
    mimeType = 'image/png'; extension = 'png'; size = pngHeader(bytes);
  } else if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    mimeType = 'image/jpeg'; extension = 'jpg'; size = jpegHeader(bytes);
  } else throw new ApiError('Only JPEG and PNG photos are supported.', 415, 'IMAGE_TYPE');
  if (declared && declared !== mimeType) throw new ApiError('Image MIME type does not match its bytes.', 415, 'IMAGE_TYPE');
  return { bytes, mimeType, extension, ...size, sha256: createHash('sha256').update(bytes).digest('hex') };
}
