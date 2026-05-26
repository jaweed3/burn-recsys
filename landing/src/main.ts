import './style.css';
import { initAnimations } from './lib/animate';
import { initCounters } from './lib/counter';
import { initNav } from './components/nav';
import { initCodeCopy } from './components/cta';

document.addEventListener('DOMContentLoaded', () => {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    // Show all animated elements immediately
    document.querySelectorAll('.animate').forEach((el) => {
      el.classList.add('visible');
    });
  } else {
    initAnimations();
    initCounters();
  }

  initNav();
  initCodeCopy();
});
