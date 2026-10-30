// A track identifies one physical mushroom, so it can occupy only one box in
// a frame. Species/group labels are separate from this temporal identity.
export function trackChoices(tracks, boxes, ownBoxId = null) {
  const used = new Set(boxes.filter(box => box.id !== ownBoxId).map(box => box.trackId).filter(Boolean));
  return tracks.map(track => ({ ...track, disabled: used.has(track.id) }));
}

export function nextBoxTrack(selectedId, boxes) {
  return selectedId && !boxes.some(box => box.trackId === selectedId) ? selectedId : null;
}

export function reviewError(status, boxes, complete) {
  if (status !== 'visible') return '';
  const tracks = new Set();
  for (const box of boxes) {
    if (!box.trackId) continue;
    if (tracks.has(box.trackId)) {
      return 'Two boxes use the same individual track. Keep it on one box; set the other to Unlinked or choose a different individual before saving.';
    }
    tracks.add(box.trackId);
  }
  if (complete && !boxes.length) return 'Draw a box around each visible mushroom, or uncheck “I boxed every visible mushroom” before saving.';
  return '';
}

export function feedAfterReview(oldFeed, currentId, filteredFeed, reviews) {
  const index = oldFeed.findIndex(frame => frame.id === currentId);
  const eligible = new Map(filteredFeed.map(frame => [frame.id, frame]));
  const following = [...oldFeed.slice(index + 1), ...oldFeed.slice(0, Math.max(0, index))];
  const next = following.find(frame => eligible.has(frame.id) && !reviews[frame.id]?.review)
    || following.find(frame => eligible.has(frame.id));
  return next ? eligible.get(next.id) : eligible.get(currentId) || filteredFeed[0] || null;
}
