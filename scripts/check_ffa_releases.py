import urllib.request, json

# Check repo info
repo_url = 'https://api.github.com/repos/zhilin007/FFA-Net'
repo = json.loads(urllib.request.urlopen(repo_url).read())
print(f"仓库: zhilin007/FFA-Net")
print(f"Stars: {repo['stargazers_count']}")
print(f"Description: {repo['description']}")
print(f"Releases: {repo['has_releases'] if 'has_releases' in repo else 'N/A'}")

# Check releases
try:
    release_url = 'https://api.github.com/repos/zhilin007/FFA-Net/releases'
    releases = json.loads(urllib.request.urlopen(release_url).read())
    print(f"\nReleases count: {len(releases)}")
    for r in releases:
        print(f"\n  Release: {r['tag_name']} - {r['name']}")
        if r['assets']:
            for a in r['assets']:
                print(f"    Asset: {a['name']} ({a['size']/1e6:.1f} MB)")
                print(f"    URL: {a['browser_download_url']}")
        else:
            print(f"    (No binary assets)")
except Exception as e:
    print(f"\nError fetching releases: {e}")

# Check if README mentions pretrained weights
try:
    readme_url = 'https://raw.githubusercontent.com/zhilin007/FFA-Net/master/README.md'
    readme = urllib.request.urlopen(readme_url).read().decode('utf-8')
    if 'pretrain' in readme.lower() or 'weight' in readme.lower():
        lines = [l for l in readme.split('\n') if 'pretrain' in l.lower() or 'weight' in l.lower() or 'model' in l.lower() or 'download' in l.lower()]
        print(f"\nREADME中相关行:")
        for l in lines[:20]:
            print(f"  {l.strip()}")
except Exception as e:
    print(f"\nError fetching README: {e}")