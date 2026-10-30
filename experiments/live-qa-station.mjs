import {createApp} from '../server.mjs';
const app=await createApp({stateFile:new URL('../data/qa-live-v8/state.json',import.meta.url).pathname.replace(/^\/([A-Z]:)/,'$1')});
await app.listen(8788);
console.log('Live workflow QA station on 8788');
