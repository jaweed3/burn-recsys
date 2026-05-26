# burn-recsys — Landing Page Design Document

> **Instruction for Code Agent:** This is the single source of truth. Build exactly what is described here. No improvisation. No "AI slop" shortcuts. Every pixel, every animation, every word is specified. Follow the design philosophy: **warm, intelligent, elegant, and quietly premium**.

---

## Table of Contents

1. [Design Philosophy & Anti-Slop Manifesto](#1-design-philosophy--anti-slop-manifesto)
2. [Color System](#2-color-system)
3. [Typography System](#3-typography-system)
4. [Layout & Spacing](#4-layout--spacing)
5. [Section 1: Navigation](#5-section-1-navigation)
6. [Section 2: Hero](#6-section-2-hero)
7. [Section 3: The Problem](#7-section-3-the-problem)
8. [Section 4: The Architecture](#8-section-4-the-architecture)
9. [Section 5: Models](#9-section-5-models)
10. [Section 6: Performance](#10-section-6-performance)
11. [Section 7: How It Works](#11-section-7-how-it-works)
12. [Section 8: Tech Stack](#12-section-8-tech-stack)
13. [Section 9: Benchmarks](#13-section-9-benchmarks)
14. [Section 10: Call to Action](#14-section-10-call-to-action)
15. [Section 11: Footer](#15-section-11-footer)
16. [Global Interactions & Animations](#16-global-interactions--animations)
17. [Responsive Behavior](#17-responsive-behavior)
18. [Performance Budget](#18-performance-budget)
19. [SEO & Meta](#19-seo--meta)
20. [File Structure](#20-file-structure)

---

## 1. Design Philosophy & Anti-Slop Manifesto

### What This Is NOT

- **No neon gradients.** No purple-to-blue radial blobs. No "AI startup circa 2023" aesthetic.
- **No glassmorphism.** No frosted panels. No backdrop-blur abuse.
- **No floating 3D shapes.** No abstract spheres, no torus knots, no "we added Three.js because we could."
- **No generic SaaS illustrations.** No flat vector people pointing at screens.
- **No dark mode cyberpunk.** Not every Rust project needs to look like a terminal emulator.
- **No loud animations.** No bouncy entrances. No spring physics on every element.

### What This IS

**Quiet confidence and thoughtful craftsmanship.** Think:

- **Aesop** meets **Stripe Press** meets **a high-end watchmaker's atelier**.
- The warmth of cream paper and aged brass. The precision of a Swiss chronometer.
- Serif headlines that whisper authority. Mono data that speaks precision.
- Every element earns its place. Every animation has a purpose.
- The feeling of reading a beautifully typeset technical journal in a quiet library.

### Core Principles

1. **Restraint over decoration.** If an element doesn't communicate value, remove it.
2. **Contrast through materiality.** Warm paper textures against cold steel data. Soft serif against sharp mono.
3. **Motion as information.** Animations guide the eye, never distract. Slow, subtle, elegant.
4. **Typography as architecture.** The type IS the design. Layout serves the type.
5. **Color as emotion.** Warmth = trust. Precision = capability. Quiet = confidence.

### The Feeling

> "Warm, intelligent, elegant, and quietly premium."

The visitor should feel like they've discovered something made by people who care deeply about craft — not a product thrown together to chase trends.

---

## 2. Color System

### Primary Palette — "Aged Paper & Brass"

| Token | Hex | Usage |
|---|---|---|
| `--bg-primary` | `#FAF7F2` | Main page background — warm white, like cream paper |
| `--bg-secondary` | `#F1E7D8` | Section alternation — soft cream, slightly deeper |
| `--bg-dark` | `#3E3128` | Dark sections (hero, footer, CTA) — deep espresso brown |
| `--surface` | `#FFFFFF` | Card backgrounds — pure white for contrast |
| `--surface-warm` | `#FDFBF7` | Warm card background — barely off-white |

### Text Colors

| Token | Hex | Usage |
|---|---|---|
| `--text-primary` | `#3E3128` | Primary body text — deep espresso brown |
| `--text-secondary` | `#8D7B68` | Secondary/muted text — warm taupe |
| `--text-light` | `#FAF7F2` | Text on dark backgrounds — cream |
| `--text-muted-light` | `#A89B8C` | Muted text on dark — warm gray |

### Accent Colors — "Brass & Precision"

| Token | Hex | Usage |
|---|---|---|
| `--accent-mocha` | `#B08968` | Primary accent — warm mocha. CTAs, links, highlights |
| `--accent-mocha-light` | `#C9A882` | Hover state — lighter mocha |
| `--accent-mocha-dark` | `#8B6B4E` | Active/pressed state — deeper mocha |
| `--accent-rust` | `#A0522D` | Secondary accent — sienna rust for emphasis |
| `--accent-forest` | `#4A7C59` | Success indicator — muted forest green |

### Border & Divider Colors

| Token | Hex | Usage |
|---|---|---|
| `--border-light` | `#E8DFD3` | Light borders — warm beige |
| `--border-dark` | `#5C4D3F` | Dark borders — warm charcoal |
| `--divider` | `#D4C8B8` | Section dividers — medium taupe |

### Usage Rules

- **Hero section:** Dark background (`--bg-dark`) with cream text. Mocha accent for CTAs.
- **Content sections:** Cream background (`--bg-primary`) with espresso text. Mocha for links and highlights.
- **Cards:** White surface (`--surface`) on cream background, OR warm surface (`--surface-warm`) on cream.
- **Alternating sections:** Odd sections use `--bg-primary`, even sections use `--bg-secondary` for gentle rhythm.
- **Borders:** 1px solid, never dashed or dotted. Subtle — they should almost disappear.
- **No gradients.** Solid colors only. The warmth comes from the palette, not from blending.

---

## 3. Typography System

### Font Families

| Role | Font | Weights | Fallback |
|---|---|---|---|
| **Display / H1-H2** | `Playfair Display` | 400 (Regular), 700 (Bold), 400i (Italic) | `Georgia, 'Times New Roman', serif` |
| **Body / UI** | `Inter` | 300 (Light), 400 (Regular), 500 (Medium), 600 (SemiBold) | `system-ui, -apple-system, sans-serif` |
| **Mono / Data / Code** | `JetBrains Mono` | 400 (Regular), 500 (Medium) | `'Fira Code', 'Courier New', monospace` |
| **Accent / Labels** | `Playfair Display Italic` | 400i | Same as Display |

**Load Strategy:**
- Playfair Display: Load only Regular, Bold, and Italic (3 weights).
- Inter: Load Light, Regular, Medium, SemiBold (4 weights).
- JetBrains Mono: Load Regular and Medium (2 weights).
- Use `font-display: swap` for all.
- **Subset:** Latin only. No Cyrillic, no extended glyphs unless needed.

### Type Scale — Fluid with `clamp()`

| Token | Mobile | Tablet | Desktop | Line Height | Letter Spacing | Usage |
|---|---|---|---|---|---|---|
| `display-xl` | `clamp(2.5rem, 7vw, 5.5rem)` | same | same | 1.05 | -0.02em | Hero headline |
| `display-lg` | `clamp(2rem, 4vw, 3.5rem)` | same | same | 1.1 | -0.015em | Section headlines |
| `display-md` | `clamp(1.5rem, 2.5vw, 2.25rem)` | same | same | 1.2 | -0.01em | Sub-headlines |
| `heading` | `clamp(1.25rem, 1.8vw, 1.75rem)` | same | same | 1.3 | -0.005em | Card titles |
| `body-lg` | `clamp(1.125rem, 1.3vw, 1.25rem)` | same | same | 1.65 | 0 | Lead paragraphs |
| `body` | `1rem` (16px) | same | same | 1.7 | 0 | Body text |
| `body-sm` | `0.875rem` (14px) | same | same | 1.6 | 0.01em | Captions, labels |
| `mono-lg` | `clamp(1rem, 1.3vw, 1.25rem)` | same | same | 1.4 | 0.02em | Large metrics |
| `mono` | `0.875rem` (14px) | same | same | 1.5 | 0.02em | Code, data |
| `mono-sm` | `0.75rem` (12px) | same | same | 1.4 | 0.03em | Small labels |
| `label` | `0.6875rem` (11px) | same | same | 1.4 | 0.1em | Uppercase labels, tracking wide |

### Typography Rules

1. **Headlines:** Playfair Display, Bold (700). Hero headline uses Italic for emphasis line to create rhythm.
2. **Body:** Inter, Regular (400). Light (300) for secondary/descriptive text.
3. **Data/Numbers:** JetBrains Mono, Medium (500). Always `font-variant-numeric: tabular-nums`.
4. **Labels/Tags:** Inter, SemiBold (600), uppercase, wide tracking (`0.1em`), small size.
5. **Accent text:** Playfair Display Italic for pull quotes, taglines, or emphasized phrases. Use sparingly.
6. **Never use more than 3 font sizes in a single section.** Hierarchy through weight and color, not size proliferation.
7. **Line length:** Max 65 characters for body text. Use `max-width: 65ch`.
8. **No ALL CAPS for headlines.** Only for labels and tags.

---

## 4. Layout & Spacing

### Grid System

- **12-column grid** on desktop, 6-column on tablet, 4-column on mobile.
- **Gutter:** 32px desktop, 24px tablet, 16px mobile.
- **Container max-width:** 1200px, centered with `margin: 0 auto`.
- **Side padding:** `clamp(1.5rem, 5vw, 4rem)` — fluid from mobile to desktop.

### Spacing Scale

| Token | Value | Usage |
|---|---|---|
| `space-1` | `0.25rem` (4px) | Micro gaps |
| `space-2` | `0.5rem` (8px) | Tight spacing |
| `space-3` | `0.75rem` (12px) | Small gaps |
| `space-4` | `1rem` (16px) | Base unit |
| `space-5` | `1.5rem` (24px) | Standard gap |
| `space-6` | `2rem` (32px) | Medium gap |
| `space-8` | `3rem` (48px) | Large gap |
| `space-10` | `4rem` (64px) | Section inner padding |
| `space-12` | `5rem` (80px) | Section separation |
| `space-16` | `6rem` (96px) | Major section break |
| `space-20` | `8rem` (128px) | Hero padding |
| `space-24` | `10rem` (160px) | Extra generous |

### Section Spacing

- **Vertical padding per section:** `clamp(5rem, 10vw, 10rem)` top and bottom.
- **Between sections:** No gap — sections touch, differentiated by background color change.
- **Content max-width within section:** Same as container (1200px).

### Border Radius

| Token | Value | Usage |
|---|---|---|
| `radius-sm` | `6px` | Buttons, small elements |
| `radius-md` | `10px` | Cards, inputs |
| `radius-lg` | `16px` | Large cards, images |
| `radius-full` | `9999px` | Pills, badges |

---

## 5. Section 1: Navigation

### Layout

- **Position:** Fixed top, `z-index: 50`.
- **Height:** 72px desktop, 64px mobile.
- **Background:** Transparent initially. On scroll (>80px): `background: rgba(250, 247, 242, 0.95)` with `backdrop-filter: blur(16px)`.
- **Border bottom on scroll:** 1px solid `--border-light`.
- **Transition:** Background and border fade in over 400ms ease.

### Content

```
[Left]                                    [Right]
┌─────────────────┐                       ┌──────────┐ ┌──────────┐
│ BR  burn-recsys │                       │ Docs     │ GitHub   │
│    (logo)       │                       │ (link)   │ (button) │
└─────────────────┘                       └──────────┘ └──────────┘
```

### Logo

- **Monogram:** "BR" in Playfair Display Bold, 18px, `--accent-mocha` color.
- **Wordmark:** "burn-recsys" in Inter SemiBold, 15px, `--text-primary`.
- **Combined:** Monogram + 10px gap + wordmark. Clickable, links to `#top`.
- **Hover:** Monogram color shifts to `--accent-mocha-light`.

### Navigation Links (Desktop)

- **Items:** "Docs" (text link), "GitHub" (outline button).
- **"Docs":** Inter Medium, 14px, `--text-secondary`. Hover: `--text-primary` with underline animation (width 0→100%, `--accent-mocha`, 250ms ease).
- **"GitHub":** Button style — see CTA buttons below. Small variant (padding 10px 20px, font 13px).

### Mobile Navigation

- **Hamburger icon:** 24px, 2px lines, `--text-primary`. Animated to X on open (300ms ease).
- **Menu overlay:** Full-screen, `--bg-primary`, fades in (opacity 0→1, 300ms).
- **Menu items:** Playfair Display, 36px, stacked vertically, centered, gap `space-6`.
- **Items:** "Architecture", "Models", "Performance", "Get Started".
- **Close:** Tap hamburger again or tap outside.

---

## 6. Section 2: Hero

### Layout

- **Height:** `100vh` minimum. `min-height: 700px`.
- **Background:** `--bg-dark` (`#3E3128`).
- **Content:** Centered vertically and horizontally. Max-width 850px for text block.
- **Padding:** `space-24` top (accounting for nav), `space-16` bottom.

### Background Effect — "Subtle Grain"

Instead of generic particles or gradients, use a **subtle noise/grain texture**:

- **Visual:** A very faint noise texture overlay at 3% opacity. Like aged paper or film grain.
- **Implementation:** CSS `background-image` with a tiny base64-encoded noise PNG, or a CSS noise pattern.
- **Color:** Warm white noise (`#FAF7F2`) at 3% opacity over the dark background.
- **No animation.** Static. The grain adds texture without movement.
- **Mobile:** Same grain, no change.

### Content Structure

```
┌─────────────────────────────────────────────────────────────┐
│  [subtle grain texture overlay — static, 3% opacity]       │
│                                                             │
│              [label]  Neural Recommendation Engine          │
│                                                             │
│     Recommendations,                                       │
│     *reimagined in Rust.*                                  │
│     ─────────────────────────────────────────────────       │
│                                                             │
│     GMF · NeuMF · DeepFM. Sub-millisecond inference.        │
│     A single static binary from training to serving.        │
│                                                             │
│     [Get Started]    [View on GitHub]                       │
│                                                             │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                       │
│  │ 0.49 │ │ 2.31M│ │ 100% │ │ <1ms │  ← live metrics       │
│  │  ms  │ │params│ │success│ │infer │                       │
│  └──────┘ └──────┘ └──────┘ └──────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Element Details

#### Label Tag
- **Text:** "Neural Recommendation Engine"
- **Style:** `label` token — Inter SemiBold, 11px, uppercase, `letter-spacing: 0.1em`, `--accent-mocha` color.
- **Animation:** Fade in + translateY(16px→0), 700ms, `cubic-bezier(0.16, 1, 0.3, 1)`, delay 200ms.

#### Headline
- **Line 1:** "Recommendations," — Playfair Display Bold, `display-xl`, `--text-light`.
- **Line 2:** "*reimagined in Rust.*" — Playfair Display Italic, `display-xl`, `--accent-mocha`.
- **Animation:** Fade in + translateY(24px→0), 900ms, same easing, delay 400ms.
- **The italic line creates the "breath" — the elegance moment.**

#### Horizontal Rule
- **Style:** 1px solid `--border-dark`, max-width 160px, centered. Margin `space-6` vertical.
- **Animation:** ScaleX(0→1) from center, 800ms, delay 600ms.

#### Subtitle
- **Text:** "GMF · NeuMF · DeepFM. Sub-millisecond inference. A single static binary from training to serving."
- **Style:** Inter Light, `body-lg`, `--text-muted-light`.
- **Animation:** Fade in + translateY(16px→0), 600ms, delay 800ms.

#### CTAs

**Primary — "Get Started"**
- Background: `--accent-mocha`. Text: `--bg-dark` (espresso), Inter SemiBold, 14px.
- Padding: 14px 32px. Border-radius: `radius-sm` (6px).
- Hover: Background → `--accent-mocha-light`, translateY(-2px), box-shadow: `0 4px 20px rgba(176, 137, 104, 0.25)`.
- Active: Background → `--accent-mocha-dark`, translateY(0).
- Transition: 250ms ease.
- Link: `#get-started`.

**Secondary — "View on GitHub"**
- Background: transparent. Border: 1px solid `--text-muted-light`. Text: `--text-light`, Inter Medium, 14px.
- Padding: 14px 32px. Border-radius: `radius-sm`.
- Hover: Border → `--accent-mocha`, text → `--accent-mocha`, translateY(-2px).
- Active: opacity 0.8.
- Link: `https://github.com/wedjaw/burn-recsys` (opens new tab).

**CTA Animation:** Fade in + translateY(16px→0), 600ms, delay 1000ms.

#### Metrics Ticker

- **Layout:** Horizontal flex, gap `space-6`, centered.
- **Each metric card:**
  - Background: `rgba(255, 255, 255, 0.04)` — barely visible.
  - Border: 1px solid `--border-dark`.
  - Border-radius: `radius-md` (10px).
  - Padding: `space-4` `space-5`.
  - Text-align: center.
  - Min-width: 100px.
- **Number:** JetBrains Mono Medium, `mono-lg`, `--accent-mocha`.
- **Label:** Inter Medium, `mono-sm`, `--text-muted-light`, uppercase, `letter-spacing: 0.05em`.
- **Metrics:**
  1. "0.49" / "ms latency"
  2. "2.31M" / "parameters"
  3. "100%" / "success rate"
  4. "<1" / "ms inference"
- **Animation:** Numbers count up from 0 to final value over 1800ms, `cubic-bezier(0.16, 1, 0.3, 1)`, triggered when scrolled into view. Stagger: 150ms between each.
- **Mobile:** 2×2 grid instead of horizontal row. Gap `space-4`.

---

## 7. Section 3: The Problem

### Layout

- **Background:** `--bg-primary` (warm white).
- **Padding:** Standard section padding.
- **Structure:** Centered text block → Three-column comparison → Bottom text.

### Content

#### Section Label
- **Text:** "The Landscape"
- **Style:** `label` token, `--accent-mocha` color, centered.
- **Animation:** Fade in on scroll.

#### Headline
- **Text:** "Python dominates recommendation systems." / "*At a steep cost.*"
- **Style:** Playfair Display Bold, `display-lg`, `--text-primary`, centered. Second line italic, `--accent-mocha`.
- **Animation:** Fade in + translateY(20px→0), 700ms.

#### Comparison Cards — Three Columns

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  PYTHON         │  │  HYBRID         │  │  BURN-RECSYS    │
│  + PYTORCH      │  │  PYTHON + RUST  │  │  (OURS)         │
│                 │  │                 │  │                 │
│  5–20ms         │  │  2–10ms         │  │  <1ms           │
│  inference      │  │  inference      │  │  inference      │
│                 │  │                 │  │                 │
│  GIL contention │  │  FFI overhead   │  │  Zero locks     │
│  GC pauses      │  │  Dual runtime   │  │  Compile-time   │
│                 │  │                 │  │  safety         │
│  $$$$$$$$       │  │  $$$$           │  │  $$             │
│  (infra cost)   │  │  (infra cost)   │  │  (infra cost)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Card 1 — Python + PyTorch:**
- Title: "Python + PyTorch" — Inter SemiBold, `heading`, `--text-primary`.
- Metric: "5–20ms" — JetBrains Mono, `mono-lg`, `--accent-rust`.
- Label: "inference" — Inter Regular, `body-sm`, `--text-secondary`.
- Issues: "GIL contention", "GC pauses" — Inter Regular, `body-sm`, `--text-secondary`, each on its own line.
- Cost: "$$$$$$$$" — JetBrains Mono, `--accent-rust`.
- Tag: "Legacy" — `label` style, background `rgba(160, 82, 45, 0.08)`, text `--accent-rust`.
- Border: 1px solid `--border-light`.
- Background: `--surface` (white).

**Card 2 — Hybrid Python + Rust:**
- Title: "Hybrid" — same style.
- Metric: "2–10ms" — JetBrains Mono, `--accent-mocha`.
- Issues: "FFI overhead", "Dual runtime" — same style.
- Cost: "$$$$" — JetBrains Mono, `--accent-mocha`.
- Tag: "Complex" — `label` style, background `rgba(176, 137, 104, 0.08)`, text `--accent-mocha`.
- Same card styling.

**Card 3 — burn-recsys (Ours):**
- Title: "burn-recsys" — same style, BUT `--accent-forest` color for title.
- Metric: "<1ms" — JetBrains Mono, `mono-lg`, `--accent-forest`.
- Issues: "Zero locks", "Compile-time safety" — same style.
- Cost: "$$" — JetBrains Mono, `--accent-forest`.
- Tag: "Optimal" — `label` style, background `rgba(74, 124, 89, 0.08)`, text `--accent-forest`.
- Border: 2px solid `--accent-forest` — slightly thicker to indicate "selected."
- Background: `--surface` with very subtle green tint: `rgba(74, 124, 89, 0.02)`.
- **This card has a subtle glow:** `box-shadow: 0 0 40px rgba(74, 124, 89, 0.06)`.

**Card Animation:** Staggered fade in + translateY(30px→0), 700ms each, 150ms stagger. Triggered on scroll.

**Card Hover (desktop):**
- translateY(-4px)
- box-shadow: `0 12px 40px rgba(62, 49, 40, 0.08)`
- Border color intensifies
- Transition: 300ms ease.

#### Bottom Text
- **Text:** "Most production recommender systems are Python-first. They work — until scale demands sub-millisecond latency, predictable memory, and deployment simplicity. Then the interpreter becomes the bottleneck."
- **Style:** Inter Light, `body-lg`, `--text-secondary`, centered, max-width 680px.
- **Animation:** Fade in, 600ms.

---

## 8. Section 4: The Architecture

### Layout

- **Background:** `--bg-secondary` (soft cream — creates gentle section separation).
- **Padding:** Standard section padding.
- **Structure:** Two-column split on desktop (45/55), stacked on mobile.

### Left Column — Text

#### Section Label
- **Text:** "Architecture"
- **Style:** `label` token, `--accent-mocha`.

#### Headline
- **Text:** "Two-stage retrieval." / "*Single binary.*"
- **Style:** Playfair Display Bold, `display-lg`, `--text-primary`. Second line italic, `--accent-mocha`.

#### Body Text
- **Paragraph 1:** "Stage 1: HNSW ANN retrieves the top-100 nearest item vectors from 7,988 candidates. Stage 2: The neural ranker scores each candidate with a sigmoid forward pass."
  - Style: Inter Regular, `body-lg`, `--text-primary`.
- **Paragraph 2:** "Worker pool architecture. One model clone per thread. Zero lock contention. mpsc dispatch."
  - Style: Inter Light, `body`, `--text-secondary`.
  - Each sentence on its own line for impact.

#### Architecture Highlights

```
✓ HNSW ANN — instant-distance, Euclidean L2
✓ Neural Ranker — sigmoid score, sort descending
✓ Worker Pool — mpsc channel, zero contention
✓ Same Binary — train, evaluate, serve
```

- **Checkmark:** "✓" in `--accent-forest`, JetBrains Mono.
- **Text:** Inter Regular, `body`, `--text-primary`.
- **Animation:** Each item fades in with 100ms stagger on scroll.

### Right Column — Pipeline Visualization

Create a **vertical flow diagram** — clean, minimal, editorial:

```
┌─────────────────────────────────────────┐
│                                         │
│    ┌─────────┐                          │
│    │  RAW    │  CSV via Polars          │
│    │  CSV    │  LazyFrame               │
│    └────┬────┘                          │
│         │                               │
│         ▼                               │
│    ┌─────────┐                          │
│    │  DATA   │  Re-index · Dedup ·      │
│    │  PIPE   │  Temporal Split          │
│    └────┬────┘                          │
│         │                               │
│         ▼                               │
│    ┌─────────┐                          │
│    │ TRAINER │  Adam · BCE · Early Stop │
│    │         │  best.mpk checkpoint     │
│    └────┬────┘                          │
│         │                               │
│         ▼                               │
│    ┌─────────┐                          │
│    │  HNSW   │  Item embeddings → graph │
│    │  INDEX  │  At server startup       │
│    └────┬────┘                          │
│         │                               │
│         ▼                               │
│    ┌─────────┐     ┌─────────┐         │
│    │  RANKER │────▶│  AXUM   │         │
│    │ Neural  │     │  HTTP   │         │
│    └─────────┘     └─────────┘         │
│                                         │
└─────────────────────────────────────────┘
```

**Node Styling:**
- Each node: Background `--surface`, border 1px solid `--border-light`, border-radius `radius-md`, padding `space-4` `space-5`.
- **Data label (e.g., "CSV"):** JetBrains Mono, `mono-sm`, `--accent-mocha`, on its own line above title.
- **Title:** Inter SemiBold, `body`, `--text-primary`.
- **Description:** Inter Regular, `body-sm`, `--text-secondary`.

**Connector Lines:**
- Vertical lines between nodes, 2px wide, `--border-light`.
- Arrow heads: Small SVG triangle, `--accent-mocha`.
- **Animation:** Lines draw from top to bottom as user scrolls (SVG stroke-dashoffset animation, 1200ms). Nodes fade in sequentially (200ms stagger).

**Container:**
- Background: `--surface` (white).
- Border: 1px solid `--border-light`.
- Border-radius: `radius-lg` (16px).
- Padding: `space-8`.
- Box-shadow: `0 4px 24px rgba(62, 49, 40, 0.04)`.

---

## 9. Section 5: Models

### Layout

- **Background:** `--bg-primary` (warm white).
- **Padding:** Standard section padding.
- **Structure:** Section header → 3 model cards in a row.

### Section Header

#### Label
- **Text:** "Models"
- **Style:** `label` token, `--accent-mocha`, centered.

#### Headline
- **Text:** "Three architectures." / "*One framework.*"
- **Style:** Playfair Display Bold, `display-lg`, `--text-primary`, centered. Second line italic, `--accent-mocha`.

#### Subtitle
- **Text:** "From matrix factorization to deep factorization machines."
- **Style:** Playfair Display Italic, `body-lg`, `--text-secondary`, centered.

### Model Cards

**Grid:** 3 columns desktop, 1 column mobile. Gap: `space-6`.

**Card Structure:**
```
┌─────────────────────────────────┐
│  [model name badge]             │
│                                 │
│  Architecture Title             │
│  Brief description              │
│                                 │
│  ─────────────────────────────  │
│  HR@10    NDCG@10    Params     │
│  0.180    0.103      1.15M      │
│                                 │
│  [architecture diagram — SVG]   │
└─────────────────────────────────┘
```

**Card Styling:**
- Background: `--surface` (white).
- Border: 1px solid `--border-light`.
- Border-radius: `radius-lg` (16px).
- Padding: `space-8`.
- Hover: translateY(-6px), border-color → `--accent-mocha`, box-shadow: `0 16px 48px rgba(62, 49, 40, 0.08)`.
- Transition: 400ms ease.

**Card 1 — GMF:**
- Badge: "GMF" — `label` style, background `rgba(176, 137, 104, 0.1)`, text `--accent-mocha`.
- Title: "Generalized Matrix Factorization" — Inter SemiBold, `heading`, `--text-primary`.
- Description: "Neural interpretation of matrix factorization. Single embedding space with element-wise product." — Inter Regular, `body-sm`, `--text-secondary`.
- Metrics row:
  - "HR@10" / "0.180" — label + value
  - "NDCG@10" / "0.103" — label + value
  - "Params" / "1.15M" — label + value
  - All values: JetBrains Mono, `mono`, `--accent-mocha`.
  - All labels: Inter Medium, `mono-sm`, `--text-secondary`, uppercase.
- Architecture: Simple SVG — two vectors (user, item) with element-wise product symbol (⊗) and arrow to output.

**Card 2 — NeuMF (Featured):**
- Badge: "NeuMF" — `label` style, background `--accent-mocha`, text `--bg-dark` (white text on mocha).
- Title: "Neural Matrix Factorization" — same style.
- Description: "Dual-path architecture. GMF path + MLP path with ReLU tower. Fusion layer concatenates both." — same style.
- Metrics:
  - HR@10: **0.604** (bold, `--accent-forest` — best)
  - NDCG@10: **0.414** (bold, `--accent-forest`)
  - Params: 2.31M
- **This card is slightly elevated:** translateY(-8px) by default, larger shadow.
- Architecture: SVG — two parallel paths (GMF and MLP) converging to fusion layer.

**Card 3 — DeepFM:**
- Badge: "DeepFM" — same as GMF badge.
- Title: "Deep Factorization Machine" — same style.
- Description: "Shared embeddings for FM and Deep paths. First-order + second-order + MLP combined." — same style.
- Metrics:
  - HR@10: "—" (dash, not yet benchmarked)
  - NDCG@10: "—"
  - Params: 1.19M
- Architecture: SVG — FM path (dot product) + Deep path (MLP) combined.

**Metrics Layout within card:**
- Horizontal flex, justify-between, border-top 1px `--border-light`, padding-top `space-4`.
- Each metric: flex-col, gap `space-1`.

**Animation:** Cards fade in with stagger (150ms each) on scroll. NeuMF card has a subtle "featured" glow pulse (box-shadow oscillates subtly, 4s loop, very gentle).

---

## 10. Section 6: Performance

### Layout

- **Background:** `--bg-dark` (`#3E3128`).
- **Padding:** Standard section padding.
- **Text color:** `--text-light` and `--text-muted-light`.
- **Structure:** Section header → Latency table → Key insight quote.

### Section Header

#### Label
- **Text:** "Performance"
- **Style:** `label` token, `--accent-mocha`.

#### Headline
- **Text:** "Sub-millisecond inference." / "*Predictably.*"
- **Style:** Playfair Display Bold, `display-lg`, `--text-light`. Second line italic, `--accent-mocha`.

### Latency Table

A clean, editorial table — not a data grid:

```
┌────────────────────────────────────────────────────────────────────┐
│  Scenario          │  Avg     │  p50     │  p90     │  p95     │  p99    │
├────────────────────────────────────────────────────────────────────┤
│  3 VUs, random     │  1.29ms  │  1.22ms  │  1.88ms  │  1.98ms  │  2.13ms │
│  With candidates   │  —       │  —       │  —       │  —       │  0.12ms │
│  Full ANN          │  —       │  —       │  —       │  —       │  0.97ms │
└────────────────────────────────────────────────────────────────────┘
```

**Table Styling:**
- Background: transparent.
- Border: 1px solid `--border-dark`.
- Border-radius: `radius-md` (10px).
- Overflow: hidden.

**Header Row:**
- Background: `rgba(255, 255, 255, 0.04)`.
- Text: Inter SemiBold, `body-sm`, `--text-muted-light`, uppercase.
- Padding: `space-4` `space-5`.

**Data Rows:**
- Text: Inter Regular, `body`, `--text-light`.
- Numbers: JetBrains Mono, `mono`, `--accent-mocha`.
- Padding: `space-4` `space-5`.
- Border-bottom: 1px solid `--border-dark` (except last row).
- Hover: Background `rgba(255, 255, 255, 0.02)`.

**Highlight:** The "0.97ms" and "0.12ms" cells have `--accent-forest` color to emphasize the speed.

**Note below table:**
- Text: "100% success rate. Zero errors across all load test requests. M4 CPU, 10 worker threads."
- Style: Inter Light, `body-sm`, `--text-muted-light`.

**Animation:** Table fades in + translateY(20px→0), 700ms. Each row has 80ms stagger.

### Key Insight Quote

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  "The ~0.5ms median inference is not from kernel            │
│   optimization — it's from eliminating the Python            │
│   interpreter layer entirely."                               │
│                                                             │
│  — Architecture Notes                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Styling:**
- Background: `rgba(255, 255, 255, 0.03)`.
- Border-left: 3px solid `--accent-mocha`.
- Padding: `space-8`.
- Quote text: Playfair Display Italic, `body-lg`, `--text-light`.
- Attribution: Inter Medium, `body-sm`, `--text-muted-light`, margin-top `space-4`.

**Animation:** Fade in, 600ms, delay 400ms after table.

---

## 11. Section 7: How It Works

### Layout

- **Background:** `--bg-primary` (warm white).
- **Padding:** Standard section padding.
- **Structure:** Section header → 4 numbered steps, full-width, alternating layout.

### Section Header

#### Label
- **Text:** "Pipeline"
- **Style:** `label` token, `--accent-mocha`, centered.

#### Headline
- **Text:** "From CSV to recommendation." / "*In seven commands.*"
- **Style:** Playfair Display Bold, `display-lg`, `--text-primary`, centered. Second line italic, `--accent-mocha`.

### Steps — Alternating Layout

**Pattern:**
- Step 1: Text left, visual right.
- Step 2: Visual left, text right.
- Step 3: Text left, visual right.
- Step 4: Visual left, text right.

**On mobile:** All stacked — number + text + visual, in that order.

#### Step Number

- **Format:** "01", "02", "03", "04" — JetBrains Mono, `display-md`, `--accent-mocha`.
- **Opacity:** 0.15 — large, decorative, behind the content.
- **Position:** Absolute, top-left of the step block, offset -30px left, font-size `clamp(4rem, 10vw, 8rem)`.

#### Step Title

- **Format:** "Load" — Inter SemiBold, `heading`, `--text-primary`.
- **Subtitle:** "Polars LazyFrame scans CSV in a single multithreaded pass." — Inter Light, `body`, `--text-secondary`.

#### Step Content

**Step 1 — LOAD:**
- Title: "Load"
- Subtitle: "Polars LazyFrame scans CSV in a single multithreaded pass. Re-indexing builds hash maps in O(n). 694K rows in <200ms."
- Visual: Minimalist line drawing of a CSV file being parsed — clean SVG, `--accent-mocha` on `--surface` background.

**Step 2 — TRAIN:**
- Title: "Train"
- Subtitle: "Adam optimizer, binary cross-entropy, early stopping by HR@k patience. Auto-saves best.mpk + config.toml. 5 epochs, ~4 minutes on M4."
- Visual: Line drawing of a loss curve descending — simple chart SVG.

**Step 3 — INDEX:**
- Title: "Index"
- Subtitle: "All item embeddings feed into an HNSW graph at server startup. instant-distance, Euclidean L2. Retrieval: ~0.12ms."
- Visual: Line drawing of a graph/network with nodes and edges — HNSW visualization.

**Step 4 — SERVE:**
- Title: "Serve"
- Subtitle: "Axum HTTP API. mpsc worker pool. One model clone per thread. POST /recommend returns ranked items in <1ms. OpenTelemetry metrics out of the box."
- Visual: Line drawing of a server with request/response arrows.

**Visual Container:**
- Background: `--surface`.
- Border: 1px solid `--border-light`.
- Border-radius: `radius-lg` (16px).
- Padding: `space-8`.
- Aspect ratio: 4:3 or 16:10.
- Box-shadow: `0 4px 24px rgba(62, 49, 40, 0.04)`.

**Step Animation:**
- Number fades in (opacity 0→0.15).
- Text slides in from its side (translateX ±30px→0, 700ms).
- Visual fades in + scale(0.97→1, 600ms).
- All triggered by scroll, staggered within step (150ms between elements).

---

## 12. Section 8: Tech Stack

### Layout

- **Background:** `--bg-secondary` (soft cream).
- **Padding:** Standard section padding.
- **Structure:** Section header → Grid of tech items.

### Section Header

#### Label
- **Text:** "Built With"
- **Style:** `label` token, `--accent-mocha`, centered.

#### Headline
- **Text:** "Technology Stack"
- **Style:** Playfair Display Bold, `display-lg`, `--text-primary`, centered.

#### Subtitle
- **Text:** "Every dependency chosen with intention."
- **Style:** Playfair Display Italic, `body-lg`, `--text-secondary`, centered.

### Tech Grid

**Layout:** Not cards. A clean, editorial grid — almost like a type specimen page.

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Deep Learning          Burn 0.17                                  │
│  ───────────────────────────────────────────────────────────────   │
│  Forward pass, backprop, model definition. Vendor-neutral.         │
│                                                                    │
│  Data Pipeline          Polars 0.46                                │
│  ───────────────────────────────────────────────────────────────   │
│  Lazy CSV loading, dedup, re-index. Single pass, zero copies.      │
│                                                                    │
│  HTTP Server            Axum 0.7                                   │
│  ───────────────────────────────────────────────────────────────   │
│  Async routing, middleware, OpenAPI docs via utoipa.               │
│                                                                    │
│  ANN Retrieval          instant-distance 0.6                       │
│  ───────────────────────────────────────────────────────────────   │
│  HNSW vector search. Euclidean L2. Logarithmic lookup time.        │
│                                                                    │
│  Observability          OpenTelemetry 0.22                         │
│  ───────────────────────────────────────────────────────────────   │
│  Metrics pipeline. Vendor-neutral. stdout → Prometheus ready.      │
│                                                                    │
│  Async Runtime          Tokio 1                                    │
│  ───────────────────────────────────────────────────────────────   │
│  Work-stealing thread pool. The foundation of everything async.    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Item Styling:**
- Category: Inter SemiBold, `body-sm`, `--text-secondary`, uppercase, `letter-spacing: 0.05em`.
- Name: Inter SemiBold, `heading`, `--text-primary`.
- Version: JetBrains Mono, `mono-sm`, `--accent-mocha`.
- Description: Inter Light, `body`, `--text-secondary`.
- Divider: 1px solid `--border-light`, full width. Margin `space-4` vertical.

**Hover (desktop):**
- Name color → `--accent-mocha`.
- Transition: 200ms.

**Animation:** Each item fades in with 80ms stagger on scroll.

---

## 13. Section 9: Benchmarks

### Layout

- **Background:** `--bg-primary` (warm white).
- **Padding:** Standard section padding.
- **Structure:** Section header → Two-column comparison (GMF vs NeuMF) → Dataset info.

### Section Header

#### Label
- **Text:** "Benchmarks"
- **Style:** `label` token, `--accent-mocha`, centered.

#### Headline
- **Text:** "Myket dataset." / "*694K interactions.*"
- **Style:** Playfair Display Bold, `display-lg`, `--text-primary`, centered. Second line italic, `--accent-mocha`.

### Comparison

```
┌─────────────────────────────┐  ┌─────────────────────────────┐
│  GMF                        │  │  NeuMF                      │
│  Generalized Matrix         │  │  Neural Matrix Factorization│
│  Factorization              │  │                             │
│                             │  │                             │
│  HR@10     0.180            │  │  HR@10     0.604            │
│  NDCG@10   0.103            │  │  NDCG@10   0.414            │
│  Params    1.15M            │  │  Params    2.31M            │
│                             │  │                             │
│  [simple bar: 30% width]    │  │  [simple bar: 100% width]   │
│                             │  │  [featured — mocha border]  │
└─────────────────────────────┘  └─────────────────────────────┘
```

**Card Styling:**
- Background: `--surface`.
- Border: 1px solid `--border-light`.
- Border-radius: `radius-lg` (16px).
- Padding: `space-8`.

**NeuMF card (featured):**
- Border: 2px solid `--accent-mocha`.
- Box-shadow: `0 0 60px rgba(176, 137, 104, 0.08)`.

**Metric display:**
- Label: Inter Medium, `body-sm`, `--text-secondary`, uppercase.
- Value: JetBrains Mono, `display-md`, `--text-primary`.
- For NeuMF HR@10 and NDCG@10: Value color is `--accent-forest` (best).

**Bar visualization:**
- Simple horizontal bar below metrics.
- Background: `--bg-secondary`.
- Fill: `--accent-mocha`.
- Height: 4px.
- Border-radius: `radius-full`.
- GMF bar: 30% width. NeuMF bar: 100% width.
- **Animation:** Bars grow from 0% to final width on scroll (800ms, ease-out).

### Dataset Info

```
┌─────────────────────────────────────────────────────────────┐
│  Dataset: Myket (Android Apps)                              │
│  Users: 10,000  ·  Items: 7,988  ·  Interactions: 694,121   │
│  Protocol: Leave-one-out temporal split  ·  Negatives: 99   │
│  Hardware: Apple M4 CPU  ·  Epochs: 5  ·  Time: ~4 min      │
└─────────────────────────────────────────────────────────────┘
```

**Styling:**
- Background: `--surface`.
- Border: 1px solid `--border-light`.
- Border-radius: `radius-md`.
- Padding: `space-6`.
- Text: Inter Regular, `body-sm`, `--text-secondary`.
- Numbers: JetBrains Mono, `mono-sm`, `--text-primary`.
- Separators: "·" in `--text-secondary`.

---

## 14. Section 10: Call to Action

### Layout

- **Background:** `--bg-dark` (`#3E3128`).
- **Padding:** `space-24` vertical — generous, this is the climax.
- **Structure:** Centered content block.

### Content

#### Headline
- **Text:** "Ready to build?"
- **Style:** Playfair Display Bold, `display-xl`, `--text-light`, centered.

#### Subtitle
- **Text:** "Seven commands from clone to recommendation. No Python runtime. No dependency hell. Just Rust."
- **Style:** Inter Light, `body-lg`, `--text-muted-light`, centered, max-width 600px.

#### CTA Buttons

**Primary — "Get Started →"**
- Background: `--accent-mocha`. Text: `--bg-dark`, Inter SemiBold, 14px.
- Padding: 14px 32px. Border-radius: `radius-sm`.
- Hover: Background → `--accent-mocha-light`, translateY(-2px), box-shadow: `0 4px 20px rgba(176, 137, 104, 0.3)`.
- Link: `#get-started`.

**Secondary — "Star on GitHub →"**
- Background: transparent. Border: 1px solid `--text-muted-light`. Text: `--text-light`, Inter Medium, 14px.
- Padding: 14px 32px. Border-radius: `radius-sm`.
- Hover: Border → `--accent-mocha`, text → `--accent-mocha`.
- Link: `https://github.com/wedjaw/burn-recsys`.

**Button Layout:** Horizontal flex, gap `space-4`, centered. Margin-top: `space-8`.

#### Code Block

```
┌─────────────────────────────────────────────────────────────┐
│  $ git clone https://github.com/wedjaw/burn-recsys          │
│  $ cd burn-recsys                                           │
│  $ uv run python scripts/download_myket.py                  │
│  $ cargo run --release --example myket_ncf                  │
│  $ cargo run --release --bin server                         │
│  $ curl -X POST http://localhost:3000/recommend \\           │
│       -H 'x-api-key: admin_bismillah' \\                     │
│       -d '{"user_id": 42}'                                  │
└─────────────────────────────────────────────────────────────┘
```

**Styling:**
- Background: `rgba(0, 0, 0, 0.2)` — subtle darkening.
- Border: 1px solid `--border-dark`.
- Border-radius: `radius-md` (10px).
- Padding: `space-6`.
- Font: JetBrains Mono, `mono`, `--text-light`.
- Max-width: 720px, centered.
- Margin-top: `space-10`.

**Copy Button:**
- Position: Absolute, top-right of code block (offset 12px from top-right corner).
- Icon: Two overlapping squares (copy), 16px, `--text-muted-light`.
- Hover: `--text-light`.
- Click: Icon changes to checkmark, "Copied" tooltip appears for 2 seconds (fade in/out).

**Animation:**
- Headline fades in + translateY(20px→0), 700ms.
- Buttons fade in, 200ms delay.
- Code block fades in + translateY(20px→0), 400ms delay.

---

## 15. Section 11: Footer

### Layout

- **Background:** `--bg-dark` (`#3E3128`).
- **Padding:** `space-12` top, `space-6` bottom.
- **Border-top:** 1px solid `--border-dark`.
- **Structure:** Three rows.

### Row 1 — Brand

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   BR  burn-recsys                                           │
│   Neural Recommendation Engine in Rust                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- Logo: Same as nav (monogram + wordmark), but larger. Monogram 22px, wordmark 16px.
- Tagline: Inter Light, `body`, `--text-muted-light`. Margin-top: `space-2`.

### Row 2 — Links & Info

```
┌────────────────────────┬────────────────────────┬──────────┐
│  Built With            │  Resources             │  Connect │
│  ───────────           │  ──────────            │  ─────── │
│  Burn 0.17             │  Documentation         │  GitHub  │
│  Polars 0.46           │  API Reference         │  Twitter │
│  Axum 0.7              │  Changelog             │  Email   │
│  OpenTelemetry         │                        │          │
└────────────────────────┴────────────────────────┴──────────┘
```

**Column Title:** Inter SemiBold, `body-sm`, `--text-light`, uppercase, `letter-spacing: 0.05em`.
**Column Items:** Inter Regular, `body-sm`, `--text-muted-light`.
**Hover:** Color → `--accent-mocha`, 200ms.

**Grid:** 3 columns desktop, 1 column mobile (stacked, gap `space-8`).

### Row 3 — Bottom Bar

```
┌─────────────────────────────────────────────────────────────┐
│  © 2026 burn-recsys  ·  MIT License  ·  Open Source        │
└─────────────────────────────────────────────────────────────┘
```

- Border-top: 1px solid `--border-dark`.
- Padding-top: `space-6`.
- Text: Inter Regular, `mono-sm`, `--text-muted-light`.

---

## 16. Global Interactions & Animations

### Scroll-Triggered Animations

Use `IntersectionObserver` with threshold 0.12 for all scroll animations.

**Default entrance animation:**
- `opacity: 0 → 1`
- `translateY: 24px → 0`
- Duration: 800ms
- Easing: `cubic-bezier(0.16, 1, 0.3, 1)` — smooth deceleration.

**Stagger pattern:**
- Multiple elements in a group: 100ms stagger between each.
- Cards in a grid: 80ms stagger.
- List items: 60ms stagger.

### Hover States

**Buttons:**
- translateY(-2px)
- Shadow increase
- Color shift
- Duration: 250ms ease.

**Cards:**
- translateY(-4px to -6px)
- Border color → `--accent-mocha`
- Shadow: `0 12px 40px rgba(62, 49, 40, 0.08)`
- Duration: 300ms ease.

**Links:**
- Underline animation: width 0→100% from left.
- Color → `--accent-mocha`.
- Duration: 250ms.

### Smooth Scroll

- `scroll-behavior: smooth` on `html`.
- Anchor links scroll smoothly to targets.

### Reduced Motion

- Respect `prefers-reduced-motion: reduce`.
- Disable all animations. Show content immediately.
- Keep hover states (they're not motion, they're state).

---

## 17. Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 640px | Single column, hamburger nav, stacked layout |
| Tablet | 640–1024px | 2-column grids, medium spacing |
| Desktop | > 1024px | Full layout, multi-column, max animations |

### Mobile-Specific Rules

1. **Navigation:** Hamburger menu. Full-screen overlay.
2. **Hero:** Smaller headline (clamp handles it). Metrics in 2×2 grid. Padding reduces.
3. **Problem cards:** Single column, stacked.
4. **Architecture:** Stacked — text above, pipeline below.
5. **Model cards:** Single column. NeuMF featured card loses default elevation (all same level).
6. **Pipeline steps:** All stacked. Visuals above text.
7. **Tech stack:** Single column list.
8. **Benchmarks:** Cards stack. Bars still animate.
9. **Metrics strip:** Not applicable (no separate metrics strip in this design).
10. **CTA:** Buttons stack vertically, full width. Code block has horizontal scroll.

### Touch Interactions

- Tap targets minimum 44×44px.
- No hover-dependent content (hover reveals should also work on tap).

---

## 18. Performance Budget

| Metric | Target |
|---|---|
| Lighthouse Performance | ≥ 95 |
| First Contentful Paint | < 1.2s |
| Largest Contentful Paint | < 2.5s |
| Total Blocking Time | < 80ms |
| Cumulative Layout Shift | < 0.05 |
| Total JS (uncompressed) | < 35KB |
| Total CSS (uncompressed) | < 18KB |
| Total page weight | < 400KB |
| Font files | < 120KB total (subsetted) |

### Optimization Rules

1. **Fonts:** Use Google Fonts with `display=swap`. Preconnect to `fonts.googleapis.com` and `fonts.gstatic.com`.
2. **Images:** No heavy images. SVG illustrations only. Optimize all SVGs.
3. **CSS:** All critical CSS inlined. Non-critical loaded async.
4. **JS:** Vanilla TypeScript. No framework. Minimal JS for scroll animations.
5. **Animations:** Use CSS transitions where possible. JS only for IntersectionObserver triggers.
6. **Static site:** Pre-rendered HTML. No hydration needed.

---

## 19. SEO & Meta

### HTML Meta Tags

```html
<title>burn-recsys — Neural Recommendation Engine in Rust</title>
<meta name="description" content="Production-grade recommendation system in Rust. GMF, NeuMF, DeepFM. Sub-millisecond inference. Two-stage retrieval + ranking. OpenTelemetry observability.">
<meta name="keywords" content="recommendation system, rust, machine learning, collaborative filtering, neural network, burn, recommender, edge ai">
<meta name="author" content="burn-recsys">
<meta name="robots" content="index, follow">
<meta name="theme-color" content="#3E3128">
```

### Open Graph

```html
<meta property="og:title" content="burn-recsys — Neural Recommendation Engine in Rust">
<meta property="og:description" content="GMF · NeuMF · DeepFM. Sub-millisecond inference. A single static binary from training to serving.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://wedjaw.github.io/burn-recsys">
<meta property="og:image" content="https://wedjaw.github.io/burn-recsys/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:site_name" content="burn-recsys">
```

### Twitter Card

```html
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="burn-recsys — Neural Recommendation Engine in Rust">
<meta name="twitter:description" content="Sub-millisecond inference. Zero Python. Single static binary.">
<meta name="twitter:image" content="https://wedjaw.github.io/burn-recsys/og-image.png">
```

### Favicon

- SVG favicon (dark background, mocha BR monogram).
- PNG fallback: 32×32, 180×180 (Apple touch).

### robots.txt

```
User-agent: *
Allow: /
Sitemap: https://wedjaw.github.io/burn-recsys/sitemap.xml
```

### sitemap.xml

Standard sitemap with single entry for landing page.

---

## 20. File Structure

```
landing/
├── index.html              # Entry point — all meta, critical CSS inline
├── vite.config.ts          # Vite config, base path for GH Pages
├── tsconfig.json           # TypeScript strict mode
├── package.json            # vite, typescript (dev deps only)
├── public/
│   ├── og-image.png        # 1200×630 Open Graph image
│   ├── favicon.svg         # SVG favicon
│   ├── favicon-32x32.png   # PNG fallback
│   └── apple-touch-icon.png # 180×180
├── src/
│   ├── main.ts             # Entry point — init all modules
│   ├── style.css           # Global styles, CSS custom properties, utilities
│   ├── lib/
│   │   ├── animate.ts      # IntersectionObserver scroll animations
│   │   └── counter.ts      # Number count-up animation
│   └── components/
│       ├── nav.ts          # Navigation + mobile menu
│       ├── hero.ts         # Hero section + metrics ticker
│       ├── problem.ts      # Problem section
│       ├── architecture.ts # Architecture section + pipeline viz
│       ├── models.ts       # Model cards
│       ├── performance.ts  # Performance table + quote
│       ├── pipeline.ts     # How It Works steps
│       ├── tech-stack.ts   # Tech stack list
│       ├── benchmarks.ts   # Benchmark comparison
│       ├── cta.ts          # Call to action + code copy
│       └── footer.ts       # Footer
├── .github/
│   └── workflows/
│       └── deploy.yml      # GitHub Actions → GitHub Pages
└── dist/                   # Build output (gitignored)
```

### Build Commands

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  }
}
```

### Vite Config

```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/burn-recsys/',  // or '/' for custom domain
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    rollupOptions: {
      output: {
        manualChunks: undefined,  // Single file, keep it simple
      },
    },
  },
});
```

---

## Appendix A: OG Image Spec

**Dimensions:** 1200×630px
**Background:** `--bg-dark` (`#3E3128`)
**Content:**
- Center: "BR" monogram in Playfair Display Bold, 120px, `--accent-mocha`.
- Below: "burn-recsys" in Inter SemiBold, 32px, `--text-light`.
- Below: "Neural Recommendation Engine in Rust" in Inter Light, 18px, `--text-muted-light`.
- Subtle grain texture overlay at 3% opacity.

**Format:** PNG, optimized (< 150KB).

---

## Appendix B: Asset Checklist

| Asset | Format | Size Limit | Source |
|---|---|---|---|
| OG Image | PNG | < 150KB | Generate per spec above |
| Favicon SVG | SVG | < 5KB | Inline monogram |
| Favicon PNG | PNG | < 10KB | Export from SVG |
| Apple Touch Icon | PNG | < 20KB | 180×180 export |
| Pipeline illustrations | SVG | < 5KB each | Hand-coded SVG |
| Architecture diagrams | SVG | < 5KB each | Hand-coded SVG |
| Model diagrams | SVG | < 3KB each | Hand-coded SVG |

---

## Appendix C: Accessibility Checklist

- [ ] All images have descriptive `alt` text.
- [ ] Color contrast ratios ≥ 4.5:1 for body text, ≥ 3:1 for large text.
- [ ] Focus states visible on all interactive elements (2px outline, `--accent-mocha`).
- [ ] Keyboard navigation works for all interactive elements.
- [ ] `prefers-reduced-motion` respected.
- [ ] Semantic HTML: `<header>`, `<main>`, `<section>`, `<footer>`, `<nav>`.
- [ ] Heading hierarchy: single H1, logical H2→H3 flow.
- [ ] ARIA labels on icon-only buttons.
- [ ] Skip-to-content link.
- [ ] Form labels (if any forms added later).

---

## Appendix D: Copy Tone Guide

**Voice:** Technical, direct, confident. Quietly proud. No fluff.

**Rules:**
1. Use concrete numbers: "0.49ms", not "very fast".
2. Use active voice: "We eliminated the Python interpreter", not "The interpreter was eliminated."
3. Short sentences. Punchy. One idea per sentence.
4. No buzzwords: No "leverage", "synergy", "paradigm shift", "disruptive", "AI-powered".
5. Technical terms are fine — the audience is engineers.
6. The tone is warm but precise. Like a colleague explaining something they built with care.

**Examples:**
- ❌ "Our solution leverages state-of-the-art deep learning architectures for next-generation recommendation systems."
- ✅ "GMF, NeuMF, DeepFM. Three architectures. One framework."

- ❌ "Experience blazing-fast inference with our cutting-edge Rust implementation."
- ✅ "Sub-millisecond inference. Predictably."

- ❌ "Cloud-native, scalable, enterprise-ready recommendation platform."
- ✅ "A single static binary from training to serving."

---

## Appendix E: Decorative Elements

### Subtle Grain Texture

Apply a static noise texture to dark sections (hero, performance, CTA, footer):

```css
.grain::before {
  content: '';
  position: absolute;
  inset: 0;
  background-image: url("data:image/svg+xml,..."); /* tiny noise pattern */
  opacity: 0.03;
  pointer-events: none;
  z-index: 1;
}
```

### Thin Divider Lines

Use 1px solid `--divider` for section separators where background doesn't change:
- Margin: `space-16` vertical.
- Width: 100% or centered at 200px max-width for decorative breaks.

### Editorial Spacing

- Paragraphs within a text block: `margin-bottom: 1.5em`.
- Between headline and body: `margin-top: space-6`.
- Between body and CTAs: `margin-top: space-8`.
- Generous whitespace is the primary decorative element.

---

> **End of Document.**
> 
> Code Agent: Build this exactly. No shortcuts. No "good enough." Every detail matters.
