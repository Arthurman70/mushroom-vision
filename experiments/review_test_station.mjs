// Isolated browser-test station. Never writes to the user's real survey or finds database.
import { createApp } from '../server.mjs';
const app=await createApp({host:'127.0.0.1',stateFile:'data/ui-review-test/state.json',findsDir:'data/ui-review-test/finds'});
await app.listen(8788);
console.log('Isolated review test station: http://127.0.0.1:8788/review.html');
