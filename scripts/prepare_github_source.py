"""Build a source-only public tree; refuse private/runtime artifacts or secrets."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('public', 'core', 'community', 'tests', 'adapters', 'android', 'docs',
           'deploy', 'scripts', 'flight', 'config', 'vision', 'experiments', 'examples', '.github')
TOP = ('README.md', 'TRAINING_PIPELINE.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE',
       'CONTRIBUTING.md', 'SECURITY.md', 'package.json', 'server.mjs',
       'START-MYCOSCOUT.cmd', '.gitignore', '.gitattributes', '.dockerignore')
SKIP_DIRS = {'data', 'dist', 'models', 'runtime-vision', 'runtime-media', 'node_modules',
             '__pycache__', '.gradle', '.git', '.venv', 'build', 'runs', 'results',
             'checkpoints', 'tmp-tests', 'assets'}
SKIP_FILES = {'local.properties', '.env', 'mushroom-vision-site.py',
              'prepare-mushroom-vision.py', 'preview-mushroom-vision.mjs',
              'test_mushroom_vision_site.py'}
EXTENSIONS = {'.py', '.mjs', '.js', '.cjs', '.md', '.json', '.html', '.css', '.svg',
              '.webmanifest', '.txt', '.ps1', '.cmd', '.bat', '.yaml', '.yml', '.toml',
              '.gradle', '.properties', '.xml', '.java', '.kt', '.kts', '.sh', '.example'}
COMMUNITY_TEMPLATES = {'deploy/community/compose.yaml', 'deploy/community/Dockerfile',
                       'deploy/community/caddy-route.txt', 'deploy/community/.gitignore'}
SYNTHETIC_EXAMPLES = {'examples/live-session-synthetic.json',
                      'examples/live-session-navigation-synthetic.json'}
PRIVATE_SCHEMAS = {'mycoscout.community-reviews.v1', 'mycoscout.community-inbox.v1',
                   'mycoscout.community-deployment-result.v1', 'mycoscout.review-export.v1',
                   'mycoscout.ai-visual-review.v1', 'mycoscout.ai-visual-review-run.v1',
                   'mycoscout.ai-visual-review-export.v1'}
SECRET_PATTERNS = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----'),
    'github-token': re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    'github-fine-grained-token': re.compile(r'\bgithub_pat_[A-Za-z0-9_]{40,}\b'),
    'cloud-access-key': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    'openai-token': re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b'),
}

SECRET_PATTERNS['private-machine-path'] = re.compile(r'[A-Z]:[/\\]+Users[/\\]+[^\s/\\]+|/(?:Users|home)/[a-zA-Z][^\s/]+', re.I)
SECRET_PATTERNS['literal-ssh-target'] = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_.-]*@(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
SECRET_PATTERNS['embedded-photographic-asset'] = re.compile(r'data:image/(?:jpeg|png|webp|gif);base64,[A-Za-z0-9+/]{128,}', re.I)


def selected(root=ROOT):
    for name in TOP:
        path = root / name
        if path.is_symlink(): raise ValueError('Linked top-level source file: '+name)
        if path.is_file(): yield path
    for folder in FOLDERS:
        if (root / folder).is_symlink(): raise ValueError('Linked source folder: '+folder)
        for base, directories, files in os.walk(root / folder, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in SKIP_DIRS
                                    and not d.startswith('tmp') and not (Path(base)/d).is_symlink())
            for name in sorted(files):
                path = Path(base)/name
                rel = path.relative_to(root).as_posix()
                if name in SKIP_FILES or name.startswith('.env.') and name != '.env.example': continue
                if name.startswith('live-session-') and rel not in SYNTHETIC_EXAMPLES: continue
                # Receipts, exports and discovery snapshots never enter a public source tree.
                if rel.startswith('deploy/community/') and path.suffix not in {'.py', '.md'} and rel not in COMMUNITY_TEMPLATES: continue
                if path.is_symlink(): raise ValueError('Linked source file: '+rel)
                if path.suffix not in EXTENSIONS and name not in {'Dockerfile', 'Caddyfile', 'gradlew', '.dockerignore', '.gitignore', 'gradle-wrapper.jar'}: continue
                if not re.fullmatch(r'[A-Za-z0-9_./-]+', rel): raise ValueError('Unexpected source filename; inspect before publishing')
                yield path

def private_json(value):
    if isinstance(value, dict):
        if value.get('schema') in PRIVATE_SCHEMAS or any(key in value for key in ('reviewerId', 'csrfToken', 'imageBase64', 'sshTarget')):
            return True
        return any(private_json(child) for child in value.values())
    if isinstance(value, list): return any(private_json(child) for child in value)
    return False


def inspect(root=ROOT):
    files = []
    findings = []
    for path in selected(root):
        raw = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        if len(raw) > 4*1024*1024: findings.append({'file':rel, 'reason':'oversized source'}); continue
        text = '' if rel == 'android/gradle/wrapper/gradle-wrapper.jar' else raw.decode('utf-8-sig')
        # Scanner test patterns are literals in this file, never credentials.
        for label, pattern in SECRET_PATTERNS.items():
            matches = list(pattern.finditer(text))
            # Loopback literals in transport rejection tests do not identify a remote host.
            if label == 'literal-ssh-target':
                matches = [m for m in matches if not m.group().split('@')[-1].startswith('127.')]
            if matches: findings.append({'file':rel, 'reason':label})
        if path.suffix == '.json':
            try:
                if private_json(json.loads(text)): findings.append({'file':rel, 'reason':'private-or-embedded-image-json'})
            except json.JSONDecodeError:
                findings.append({'file':rel, 'reason':'malformed-json'})
        files.append({'path':rel, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()})
    return files, findings

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    files, findings = inspect()
    report = {'files':len(files), 'bytes':sum(f['bytes'] for f in files), 'findings':findings,
              'sourceSelectionVerified':not findings,
              'photosIncluded':False if not findings else None,
              'modelWeightsIncluded':False if not findings else None,
              'privateReviewsIncluded':False if not findings else None}
    if findings:
        print(json.dumps(report, indent=2)); raise SystemExit(1)
    if args.output:
        output = args.output.resolve()
        if output.exists(): raise ValueError('Choose a new empty source destination; existing trees are never replaced')
        if output == ROOT or ROOT.is_relative_to(output): raise ValueError('Output cannot contain the working project')
        output.mkdir(parents=True)
        for row in files:
            target = output/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT/row['path']
            if source.is_symlink() or not source.resolve().is_relative_to(ROOT): raise ValueError('Source changed to a link during packaging')
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['sha256']: raise ValueError('Source changed during packaging; regenerate the release')
            target.write_bytes(raw)
        (output/'SOURCE_MANIFEST.json').write_text(json.dumps({'schema':'mushroom-vision.source-release.v1',
            **report, 'sourceFiles':files},indent=2)+'\n',encoding='utf8')
        report['output'] = str(output)
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
