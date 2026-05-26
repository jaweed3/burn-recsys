interface CounterOptions {
  to: number;
  decimals: number;
  suffix: string;
}

function parseOptions(el: HTMLElement): CounterOptions {
  return {
    to: parseFloat(el.dataset.countTo || '0'),
    decimals: parseInt(el.dataset.countDecimals || '0', 10),
    suffix: el.dataset.countSuffix || '',
  };
}

function animateValue(
  el: HTMLElement,
  start: number,
  end: number,
  duration: number,
  decimals: number,
  suffix: string
): void {
  const startTime = performance.now();
  const easing = (t: number) => {
    return 1 - Math.pow(1 - t, 3);
  };

  function update(currentTime: number) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easedProgress = easing(progress);
    const current = start + (end - start) * easedProgress;

    el.textContent = current.toFixed(decimals) + suffix;

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

export function initCounters(): void {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const card = entry.target as HTMLElement;
          const valueEl = card.querySelector('.metric-card__value') as HTMLElement;
          if (!valueEl) continue;

          const opts = parseOptions(card);
          const duration = 1800;
          const stagger = 150;

          const index = Array.from(
            card.parentElement?.children || []
          ).indexOf(card);

          setTimeout(() => {
            animateValue(
              valueEl,
              0,
              opts.to,
              duration,
              opts.decimals,
              opts.suffix
            );
          }, index * stagger);

          observer.unobserve(card);
        }
      }
    },
    { threshold: 0.5 }
  );

  document.querySelectorAll('[data-count-to]').forEach((el) => {
    observer.observe(el);
  });
}
