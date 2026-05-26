export function initNav(): void {
  const nav = document.getElementById('nav')!;
  const hamburger = document.querySelector('.nav__hamburger') as HTMLElement;
  const overlay = document.getElementById('nav-overlay')!;
  const overlayLinks = overlay.querySelectorAll('.nav-overlay__link');

  // Scroll handler
  window.addEventListener('scroll', () => {
    if (window.scrollY > 80) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  }, { passive: true });

  // Hamburger toggle
  hamburger.addEventListener('click', () => {
    const isOpen = hamburger.classList.toggle('open');
    overlay.classList.toggle('open', isOpen);
    hamburger.setAttribute('aria-expanded', String(isOpen));
    document.body.style.overflow = isOpen ? 'hidden' : '';
  });

  // Close on link click
  overlayLinks.forEach((link) => {
    link.addEventListener('click', () => {
      hamburger.classList.remove('open');
      overlay.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
    });
  });

  // Close on overlay click (outside nav)
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      hamburger.classList.remove('open');
      overlay.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
    }
  });
}
