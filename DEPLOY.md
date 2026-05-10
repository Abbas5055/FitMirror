# Deploying FitMirror to Hugging Face Spaces

You're shipping this to the public today. Follow these steps top-to-bottom —
do not skip any.

---

## 1. Create a Hugging Face account (2 min)

1. Go to https://huggingface.co/join
2. Sign up with the same email you use for GitHub (`mightyabbu07@gmail.com`).
3. Pick a username — recommended: `abbas5055` (matches your GitHub handle).
4. Verify the email.

## 2. Create a write token (1 min)

1. Click your avatar → **Settings** → **Access Tokens** → **New token**.
2. Name: `fitmirror-deploy`. Role: **Write**. Click **Generate**.
3. Copy the token (starts with `hf_`). Keep it open in a tab — you'll paste
   it once.

## 3. Create the Space (1 min)

1. Go to https://huggingface.co/new-space
2. Owner: your username. Space name: `FitMirror`.
3. License: **MIT**. SDK: **Gradio**. Hardware: **CPU basic (free)**.
4. Visibility: **Public**.
5. Click **Create Space**. You'll land on an empty repo page.

## 4. Push the code (3 min)

Open a terminal in your project folder (Windows PowerShell or WSL, either
works — these commands are identical):

```bash
cd "C:\Users\Abbas\Documents\FitMirror_v2\FitMirror"

# Initialise git in this folder (if you haven't already).
git init -b main
git add .
git commit -m "FitMirror v1: pose -> measurements -> Indian-wear sizing"

# Add the Hugging Face Space as a remote and push.
# Replace <USERNAME> with your HF username (e.g. abbas5055).
git remote add hf https://huggingface.co/spaces/<USERNAME>/FitMirror
git push hf main
```

When git prompts for credentials:
- **Username:** your HF username
- **Password:** paste the token from step 2 (NOT your HF password)

## 5. Wait for the build (3-5 min)

After the push completes:
1. Open `https://huggingface.co/spaces/<USERNAME>/FitMirror` in your browser.
2. You'll see a yellow **Building** banner. Click **Logs** to follow along.
3. The build runs `pip install -r requirements.txt` and then launches `app.py`.
4. When the banner turns green and the Gradio UI shows up, you're live.

If the build fails:
- Most common cause: a typo in `requirements.txt` or `app.py`. Check the logs,
  fix locally, `git commit -am "fix"`, `git push hf main`. Spaces auto-rebuilds.

## 6. Smoke test the live URL (2 min)

1. Open the live URL on your **phone** (the recruiter probably will too).
2. Upload a front-facing full-body photo of yourself.
3. Enter your real height. Pick "Male" + "Men's Kurta".
4. Click **Measure & recommend size**.
5. Confirm:
   - Pose overlay shows up on the right (skeleton drawn on you).
   - Measurements table populates with sensible numbers.
   - Size recommendation appears.
6. Try one bad input (e.g. upload a landscape photo of nothing) — confirm the
   error message is human-readable, not a stack trace.

## 7. Push the same code to GitHub (1 min)

So the recruiter can also see the source:

```bash
git remote add origin https://github.com/Abbas5055/FitMirror.git
git push -u origin main
```

(Assumes you've created the empty repo on GitHub at
https://github.com/new — name `FitMirror`, public, no README.)

---

## Total time: ~15 minutes

Once both URLs are live (`huggingface.co/spaces/...` and `github.com/...`),
you're ready to send the recruiter message (see `RECRUITER_MESSAGE.md`).
