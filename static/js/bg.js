(function () {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return; // Defensive: don't run if canvas is missing

  const ctx = canvas.getContext('2d');
  let width = window.innerWidth;
  let height = window.innerHeight;
  let particles = [];
  const PARTICLE_COUNT = 140;
  const LINE_DISTANCE = 120;

  function resizeCanvas() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
  }

  class Particle {
    constructor() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.size = Math.random() * 2.5 + 1.5;
      this.speedX = Math.random() * 0.8 - 0.4;
      this.speedY = Math.random() * 0.8 - 0.4;
      this.color = `rgba(79,195,247,${Math.random() * 0.5 + 0.4})`;
    }
    update() {
      this.x += this.speedX;
      this.y += this.speedY;
      // Bounce from edges
      if (this.x < 0 || this.x > width) this.speedX *= -1;
      if (this.y < 0 || this.y > height) this.speedY *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.shadowColor = "#4fc3f7";
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0; // Reset shadow for other drawings
    }
  }

  function initParticles() {
    particles = [];
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push(new Particle());
    }
  }

  function drawLines() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < LINE_DISTANCE) {
          ctx.save();
          ctx.globalAlpha = (LINE_DISTANCE - dist) / LINE_DISTANCE * 0.35;
          ctx.strokeStyle = "#4fc3f7";
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
          ctx.restore();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);
    for (const p of particles) {
      p.update();
      p.draw();
    }
    drawLines();
    requestAnimationFrame(animate);
  }

  // Responsive resize
  window.addEventListener('resize', () => {
    resizeCanvas();
    initParticles();
  });

  // Init
  resizeCanvas();
  initParticles();
  animate();
})();
