document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('form');
  if (!form) return;

  form.addEventListener('submit', e => {
    const age = document.querySelector('#age');
    if (age && (age.value < 14 || age.value > 65)) {
      alert('Age must be between 14 and 65');
      e.preventDefault();
    }
  });
});