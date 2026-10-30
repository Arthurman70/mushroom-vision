const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs/promises');
const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const BASE = process.env.MYCOSCOUT_TEST_URL || 'http://127.0.0.1:8787';
const executablePath = process.env.PLAYWRIGHT_BROWSER ||
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const results = { passed: 0, failed: 0, cases: [], errors: [], testedAt: new Date().toISOString(), hardwareTested: false };

function passed(name) {
  results.passed++;
  results.cases.push({ name, status: 'passed' });
  console.log('PASS ' + name);
}
function watch(page, label) {
  page.on('pageerror', error => results.errors.push(label + ': ' + error.message));
}
async function contextPage(browser, label, viewport = { width: 1440, height: 1100 }) {
  // A service worker must not bypass route interception in regression tests.
  const context = await browser.newContext({ viewport, serviceWorkers: 'block' });
  const page = await context.newPage();
  watch(page, label);
  return { context, page };
}
async function hideToast(page) {
  await page.locator('#toast').evaluate(element => { element.style.display = 'none'; });
}
function survey(verified = true) {
  return {
    schemaVersion: 1, mode: 'survey', frame: { id: 'map', unit: 'm' },
    groundTraversabilityVerified: verified,
    grid: { width: 8, height: 8, resolution: 1, origin: { x: 0, y: 0 }, cells: Array(64).fill(0) },
    observations: [{ id: 'synthetic-regression-point', x: 5.5, y: 5.5, z: 0,
      label: 'Synthetic test observation', confidence: 0, source: 'automated-test-only',
      reviewed: false, frameId: 'map', poseKnown: true, depthKnown: true }],
    hazards: [], route: null, telemetry: { connected: false, position: null, source: 'none' }, capabilities: {},
  };
}
async function importSurvey(page, value) {
  await page.locator('#survey-file').setInputFiles({
    name: 'synthetic-survey.json', mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(value)),
  });
  await page.waitForFunction(() => document.getElementById('connection').textContent === 'Offline survey');
  await page.locator('.observation').first().click();
}

(async () => {
  await fs.mkdir(DIST, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath });
  try {
    const demo = JSON.parse(await fs.readFile(path.join(ROOT, 'public', 'demo-state.json'), 'utf8'));
    const { context, page } = await contextPage(browser, 'primary');
    await page.goto(BASE);
    await page.waitForSelector('.observation');
    assert.equal(await page.locator('.observation').count(), 4);
    await page.locator('.observation').first().click();
    await page.locator('#route-plan').click();
    await page.waitForFunction(() => document.getElementById('route-status').textContent.includes('Demo route'));
    await hideToast(page);
    await page.screenshot({ path: path.join(DIST, 'desktop-preview.png'), fullPage: true });
    passed('desktop map, candidate selection, live server route');

    await page.setViewportSize({ width: 393, height: 852 });
    await hideToast(page);
    await page.screenshot({ path: path.join(DIST, 'phone-preview.png'), fullPage: true });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    passed('mobile layout without horizontal overflow');

    await page.locator('#settings-open').click();
    await page.locator('#offline-demo').click();
    await page.waitForFunction(() => document.getElementById('connection').textContent === 'Offline demo');
    await page.locator('.observation').first().click();
    await page.locator('#route-plan').click();
    await page.waitForFunction(() => document.getElementById('route-status').textContent.includes('Demo route'));
    passed('offline routing uses shared planner');

    await page.locator('#detections-file').setInputFiles({
      name: 'detections.json', mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify({ detections: [{ label: '<img src=x onerror=alert(1)>', score: 0.8 }] })),
    });
    await page.waitForFunction(() => document.getElementById('unlocated').textContent.includes('unlocated'));
    assert.equal(await page.locator('#unlocated img').count(), 0);
    assert((await page.locator('#unlocated').innerText()).includes('<img src=x onerror=alert(1)>'));
    passed('image candidate import stays unlocated and escapes HTML');

    await importSurvey(page, survey(true));
    assert.equal(await page.locator('#obstacle-label').innerText(), 'Occupied');
    await page.locator('#route-plan').click();
    await page.waitForFunction(() => document.getElementById('route-status').textContent.includes('Enter your current coordinates'));
    assert(!(await page.locator('#route-status').innerText()).includes('Advisory route'));
    await page.locator('#start-x').fill('2.5');
    await page.locator('#start-y').fill('2.5');
    await page.locator('#route-plan').click();
    await page.waitForFunction(() => document.getElementById('route-status').textContent.includes('Advisory route'));
    assert.match(await page.locator('#route-status').innerText(), /4\.2 m/);
    passed('reviewed survey import requires map-frame start and then computes a route');

    await importSurvey(page, survey(false));
    await page.locator('#start-x').fill('2.5');
    await page.locator('#start-y').fill('2.5');
    await page.locator('#route-plan').click();
    await page.waitForFunction(() => document.getElementById('route-status').textContent.includes('Ground review is required'));
    assert(!(await page.locator('#toast').innerText()).includes('Route avoids mapped hazards'));
    passed('occupancy-only survey import cannot enable walking routes');
    await context.close();

    const unavailable = await contextPage(browser, 'unavailable-route');
    await unavailable.context.route('**/api/state', route => route.fulfill({ json: demo }));
    await unavailable.context.route('**/api/route', route => route.fulfill({
      json: { route: { mode: 'walk', status: 'unavailable', reason: 'No traversable corridor at this clearance.', points: [], distance: 0 } },
    }));
    await unavailable.page.goto(BASE);
    await unavailable.page.waitForSelector('.observation');
    await unavailable.page.locator('.observation').first().click();
    await unavailable.page.locator('#route-plan').click();
    await unavailable.page.waitForFunction(() => document.getElementById('route-status').textContent.includes('No traversable corridor'));
    assert.match(await unavailable.page.locator('#route-status').innerText(), /^Route unavailable:/);
    assert.equal(await unavailable.page.locator('#toast').innerText(), 'No traversable corridor at this clearance.');
    passed('HTTP-success unavailable route is not displayed as route success');
    await unavailable.context.close();

    const empty = await contextPage(browser, 'null-map');
    const emptySurvey = { ...survey(false), grid: null, observations: [] };
    await empty.context.route('**/api/state', route => route.fulfill({ json: emptySurvey }));
    await empty.page.goto(BASE);
    await empty.page.waitForFunction(() => document.getElementById('connection').textContent === 'Station connected');
    assert.equal(await empty.page.locator('.observation').count(), 0);
    assert((await empty.page.locator('#observations').innerText()).includes('No mapped candidates'));
    await empty.page.locator('#view-top').click();
    await empty.page.locator('#view-perspective').click();
    assert.equal(await empty.page.locator('#route-plan').count(), 0);
    passed('empty survey with null grid renders without map assumptions');
    await empty.context.close();

    const disconnected = await contextPage(browser, 'unreachable-station');
    await disconnected.context.addInitScript(() => {
      localStorage.setItem('mycoscout-station', 'https://station.invalid');
    });
    await disconnected.context.route('https://station.invalid/**', route => route.abort('connectionrefused'));
    await disconnected.page.goto(BASE);
    await disconnected.page.waitForFunction(() => document.getElementById('mission-detail').textContent.includes('Connection failed'));
    const unexpectedDownload = disconnected.page.waitForEvent('download', { timeout: 600 }).then(() => true, () => false);
    await disconnected.page.locator('#export').click();
    await disconnected.page.waitForFunction(() => document.getElementById('toast').textContent === 'Connect a station or open the offline demo first.');
    assert.equal(await unexpectedDownload, false);
    assert.equal(await disconnected.page.locator('.observation').count(), 0);
    passed('unreachable remembered station cannot export undefined survey data');
    await disconnected.context.close();

    assert.equal(results.errors.length, 0, results.errors.join('\n'));
    passed('no browser script exceptions across all regression scenarios');
  } catch (error) {
    results.failed = 1;
    results.failure = error.stack || String(error);
    throw error;
  } finally {
    await browser.close();
    await fs.writeFile(path.join(DIST, 'ui-test-results.json'), JSON.stringify(results, null, 2));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
