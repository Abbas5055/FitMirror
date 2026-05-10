# Recruiter follow-up — ready to paste

Send this **after** the live URL is up and you've personally tested it on
your phone. Replace `<HF_URL>` with the actual Space URL and `<GH_URL>` with
the GitHub repo URL.

---

## Option A — WhatsApp / SMS (short)

> Hi <name>, following up on our conversation yesterday about the CV/3D role
> at Sthyra. I built a small demo over the weekend to show how I'd approach
> Stage 1 of the body-measurement pipeline for custom Indian wear.
>
> Live demo (works on mobile): <HF_URL>
> Source: <GH_URL>
>
> Single front-facing photo + height → 33-landmark pose detection → linear
> measurements + chest/waist/hip circumferences (silhouette width × ellipse
> approximation) → size recommendation across kurta, anarkali, and saree
> blouse charts. Stage 1 runs CPU-only on the free Hugging Face tier. README
> documents the accuracy bounds (±2-3 cm linear, ±4-6 cm circumference) and
> what Stage 2 (depth + SMPL) would unlock.
>
> Happy to walk through the code or the next-stage plan whenever works for
> you and the team.

---

## Option B — Email (slightly longer, same content)

**Subject:** Sthyra CV/3D role — small demo I put together

> Hi <name>,
>
> Thanks again for the conversation yesterday. To make the next round more
> concrete, I built a small Stage-1 demo of the kind of body-measurement
> pipeline that would sit in front of a custom-fit Indian-wear product:
>
> **Live demo:** <HF_URL>
> **Source:** <GH_URL>
>
> The flow:
> 1. User uploads one front-facing full-body photo and enters their height.
> 2. MediaPipe Pose extracts 33 landmarks plus a body segmentation mask.
> 3. Height calibrates pixels-to-cm.
> 4. Linear measurements (shoulder, sleeve, torso, inseam) come from
>    landmark distances scaled by the calibration.
> 5. Circumferences combine the silhouette width at the chest/waist/hip
>    line with anthropometric depth ratios (NHANES + ISI Calcutta,
>    gender-adjusted) and the Ramanujan-II ellipse perimeter formula.
> 6. The result maps to standard Indian-wear charts for men's kurta,
>    women's kurta/anarkali, and saree blouse.
>
> Honest accuracy on Stage 1 is **±2-3 cm linear, ±4-6 cm circumference**
> (documented in the README and surfaced in the UI — I didn't want to
> over-claim). Stage 2 would integrate Depth Anything v2 for real depth and
> SMPL body fitting to bring circumferences into the ±2-3 cm band, which is
> where it would actually be useful for tailoring.
>
> Everything runs CPU-only on the free Hugging Face Spaces tier, so anyone
> can try it from their phone.
>
> Happy to walk through the code or the Stage-2 plan whenever works for you
> and the engineering team.
>
> Best,
> Abbas

---

## What NOT to say

- Don't promise tailoring-grade accuracy. The demo is honest; the message
  should be too.
- Don't claim it works on every photo. Edge cases (occluded body, sideways
  pose, complex background) return a clean error message — that's a feature,
  not a defect, but don't over-sell it.
- Don't say "production-ready." Say "Stage 1 demo, deployed, public."

## What to lean into

- **Speed of execution** — interviewed yesterday, shipped a working public
  URL within 48 hours.
- **Honest engineering** — accuracy bounds are measured (or marked as n=1),
  not invented.
- **Clear roadmap** — you already know what Stage 2 looks like.
