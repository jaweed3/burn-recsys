export function initCodeCopy(): void {
  const copyBtn = document.querySelector('.cta__code-copy') as HTMLElement;
  const tooltip = copyBtn?.querySelector('.cta__code-tooltip') as HTMLElement;
  const codeBlock = document.querySelector('.cta__code code');

  if (!copyBtn || !codeBlock) return;

  copyBtn.addEventListener('click', async () => {
    const text = codeBlock.textContent || '';
    try {
      await navigator.clipboard.writeText(text.trim());
      tooltip.classList.add('show');
      setTimeout(() => {
        tooltip.classList.remove('show');
      }, 2000);
    } catch {
      // fallback
    }
  });
}
