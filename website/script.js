const revealElements = document.querySelectorAll('.reveal');
const counterElements = document.querySelectorAll('[data-count]');

const formatNumber = (value, originalTarget) => {
  const hasDecimal = String(originalTarget).includes('.');
  if (hasDecimal) {
    return Number(value).toFixed(2);
  }
  return Number(Math.round(value)).toLocaleString();
};

const runCounter = (element) => {
  if (element.dataset.counted === 'true') return;

  const target = Number(element.dataset.count);
  if (Number.isNaN(target)) return;

  element.dataset.counted = 'true';
  const durationMs = 1100;
  const start = performance.now();

  const tick = (now) => {
    const progress = Math.min((now - start) / durationMs, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = target * eased;
    element.textContent = formatNumber(current, element.dataset.count);

    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      element.textContent = formatNumber(target, element.dataset.count);
    }
  };

  requestAnimationFrame(tick);
};

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;

      entry.target.classList.add('is-visible');
      if (entry.target.matches('[data-count]')) {
        runCounter(entry.target);
      }
    });
  },
  {
    threshold: 0.15,
    rootMargin: '0px 0px -10% 0px',
  }
);

revealElements.forEach((element) => observer.observe(element));
counterElements.forEach((element) => observer.observe(element));

window.addEventListener('scroll', () => {
  const offset = window.scrollY * 0.08;
  document.querySelectorAll('.orb').forEach((orb, index) => {
    const direction = index % 2 === 0 ? 1 : -1;
    orb.style.transform = `translateY(${direction * offset}px)`;
  });
});
