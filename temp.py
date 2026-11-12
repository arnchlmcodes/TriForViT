import os, json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
ds = "defactodataset/defactoinpainting"
N = 5000
OUT = "defacto_subset"

# folders to download, and the extensions to try for each
FOLDERS = {
    "img":        "inpainting_img/img",
    "probe_mask": "inpainting_annotations/probe_mask",
    # "inpaint_mask": "inpainting_annotations/inpaint_mask",
}
EXTS = [".tif", ".jpg", ".png", ".jpeg", ".bmp"]
GRAPH = "inpainting_annotations/graph/"

# 1) get the first N IDs from graph, stopping the listing early
if not os.path.exists("ids.json"):
    ids, token, done = [], None, False
    with tqdm(total=N, desc="Collecting IDs", unit=" ids") as bar:
        while not done:
            res = api.dataset_list_files(ds, page_token=token, page_size=200)
            for f in res.files:
                if f.name.startswith(GRAPH):
                    ids.append(os.path.splitext(os.path.basename(f.name))[0])
                    bar.update(1)
                    if len(ids) >= N:
                        done = True
                        break
                elif ids:          # left the graph folder
                    done = True
                    break
            token = res.next_page_token
            if not token:
                break
    json.dump(ids, open("ids.json", "w"))
ids = json.load(open("ids.json"))
print("IDs:", len(ids))

# 2) find the working extension for each folder using the first ID
def exists_locally(folder_out, fname):
    return os.path.exists(os.path.join(folder_out, fname)) or os.path.exists(os.path.join(folder_out, fname + ".zip"))

ext_for = {}
for k, prefix in FOLDERS.items():
    out = os.path.join(OUT, k); os.makedirs(out, exist_ok=True)
    for e in EXTS:
        fname = ids[0] + e
        try:
            api.dataset_download_file(ds, f"{prefix}/{fname}", path=out, quiet=True)
        except Exception:
            continue
        if exists_locally(out, fname):
            ext_for[k] = e
            break
    print(k, "->", ext_for.get(k, "NOT FOUND"))
if len(ext_for) != len(FOLDERS):
    raise SystemExit("Couldn't find the extension for some folder; paste me a few filenames from it.")

# 3) download everything with a progress bar (skips existing files)
def fetch(args):
    k, s = args
    out = os.path.join(OUT, k)
    fname = s + ext_for[k]
    if exists_locally(out, fname):
        return "skipped"
    try:
        api.dataset_download_file(ds, f"{FOLDERS[k]}/{fname}", path=out, quiet=True)
        return "ok"
    except Exception as e:
        return f"failed: {FOLDERS[k]}/{fname} ({e})"

jobs = [(k, s) for s in ids for k in FOLDERS]
with ThreadPoolExecutor(8) as ex:
    results = list(tqdm(ex.map(fetch, jobs), total=len(jobs), desc="Downloading", unit=" file"))

failed = [r for r in results if r.startswith("failed")]
print(f"done: {results.count('ok')} downloaded, {results.count('skipped')} skipped, {len(failed)} failed")
for f in failed[:10]:
    print(f)