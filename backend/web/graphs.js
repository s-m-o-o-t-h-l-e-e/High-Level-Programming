async function loadGallery() {
  const response = await fetch(`/graphs/list?client=${Date.now()}`);
  const figures = await response.json();
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = figures.length ? figures.map(figure => `
    <article class="graph-row" style="display:grid;grid-template-columns:220px 1fr auto;gap:16px;align-items:center;">
      <a href="${figure.detail_url}"><img src="${figure.url}" alt="${figure.title}" style="width:220px;aspect-ratio:16/9;object-fit:cover;border:1px solid #d7dee8;border-radius:6px;background:#fff;" /></a>
      <div>
        <span class="eyebrow">FIGURE</span>
        <h2 style="margin:.25rem 0;">${figure.title}</h2>
        <p style="margin:0;color:#64748b;font-weight:750;">${figure.filename}</p>
      </div>
      <nav style="display:flex;gap:8px;">
        <a class="link-button" href="${figure.detail_url}">열기</a>
        <a class="link-button" href="${figure.url}">PNG</a>
      </nav>
    </article>
  `).join('') : '<p class="status-line">아직 생성된 그래프가 없습니다. /refresh를 먼저 실행하세요.</p>';
}

loadGallery();
