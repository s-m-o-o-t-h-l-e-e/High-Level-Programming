const won = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 1 });

function number(value, suffix = '') {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${won.format(parsed)}${suffix}` : '-';
}

function setText(id, text) {
  const element = document.getElementById(id);
  if (element) element.textContent = text;
}

function formatDateTime(value) {
  if (!value) return '-';
  const text = String(value).trim();
  const match = text.match(/^(\d{4}-\d{2}-\d{2})(?:[T\s](\d{2}:\d{2}(?::\d{2})?))?/);
  if (!match) return text;
  return match[2] ? `${match[1]} ${match[2]}` : match[1];
}

function changeClass(value) {
  if (value > 0.05) return 'up';
  if (value < -0.05) return 'down';
  return 'flat';
}

function changeLabel(value) {
  const sign = value > 0 ? '+' : '';
  return `${sign}${won.format(value)}`;
}

function totalForecastChange(todayPrice, forecast) {
  if (!Number.isFinite(todayPrice) || !forecast.length) return NaN;
  const lastPrice = Number(forecast[forecast.length - 1].predicted_domestic_price);
  return Number.isFinite(lastPrice) ? lastPrice - todayPrice : NaN;
}

function cacheBust(url) {
  const glue = url.includes('?') ? '&' : '?';
  return `${url}${glue}v=${Date.now()}`;
}

async function loadSummary() {
  const response = await fetch('/summary', { cache: 'no-store' });
  if (!response.ok) throw new Error(`summary ${response.status}`);
  return response.json();
}

async function loadGraphs() {
  const response = await fetch('/graphs/list', { cache: 'no-store' });
  if (!response.ok) throw new Error(`graphs ${response.status}`);
  return response.json();
}

function renderSummary(data, statusMessage = null) {
  const latest = data.latest || {};
  const forecast = data.forecast || [];
  const news = data.news || {};
  const todayPrice = Number(latest.domestic_price);
  const updatedAt = formatDateTime(latest.updated_at || latest.date);

  setText('todayDate', `${updatedAt} 기준`);
  setText('todayPrice', number(todayPrice, ' 원/L'));
  setText('wtiValue', number(latest.wti, ' $/bbl'));
  setText('brentValue', number(latest.brent, ' $/bbl'));
  setText('exchangeValue', number(latest.exchange, ' 원'));
  setText('riskValue', Number.isFinite(Number(news.news_risk_score)) ? Number(news.news_risk_score).toFixed(3) : '-');
  setText('riskMeta', `${news.article_count || latest.news_article_count || 100}개 기사`);

  const total = totalForecastChange(todayPrice, forecast);
  if (Number.isFinite(total)) {
    const direction = total >= 0 ? '상승' : '하락';
    const cls = changeClass(total);
    const summary = `현재 대비 7일 누적 ${Math.abs(total).toFixed(1)} 원/L ${direction} 전망`;
    const element = document.getElementById('sevenDaySummary');
    element.textContent = summary;
    element.className = cls;
  }

  setText('dataStatus', statusMessage || `데이터 기준일: ${updatedAt}`);
  renderForecastRows(todayPrice, forecast);
  return updatedAt;
}

function renderForecastRows(todayPrice, forecast) {
  const tbody = document.getElementById('forecastRows');
  if (!forecast.length) {
    tbody.innerHTML = '<tr><td colspan="4">예측 데이터가 없습니다.</td></tr>';
    return;
  }
  let previousPrice = todayPrice;
  tbody.innerHTML = forecast.map(row => {
    const price = Number(row.predicted_domestic_price);
    const diff = Number.isFinite(previousPrice) ? price - previousPrice : NaN;
    previousPrice = price;
    return `
      <tr>
        <td>${row.date}</td>
        <td><strong>${number(price)}</strong></td>
        <td class="${changeClass(diff)}">${changeLabel(diff)}</td>
        <td class="reason-cell">${row.reason || '-'}</td>
      </tr>
    `;
  }).join('');
}

function renderGraphs(graphs) {
  const list = document.getElementById('graphList');
  if (!graphs.length) {
    list.innerHTML = '<p class="hero-copy">그래프가 없습니다. 최신화를 먼저 실행하세요.</p>';
    return;
  }

  function selectGraph(graph) {
    document.querySelectorAll('#graphList button').forEach(button => {
      button.classList.toggle('active', button.dataset.filename === graph.filename);
    });
    setText('graphTitle', graph.title);
    const image = document.getElementById('graphImage');
    const open = document.getElementById('graphOpen');
    image.src = graph.url;
    image.alt = graph.title;
    open.href = graph.detail_url;
  }

  list.innerHTML = graphs.map((graph, index) => `
    <button class="${index === 0 ? 'active' : ''}" data-filename="${graph.filename}">
      ${graph.title}
    </button>
  `).join('');

  list.querySelectorAll('button').forEach((button, index) => {
    button.addEventListener('click', () => selectGraph(graphs[index]));
  });
  selectGraph(graphs[0]);
}

async function refreshData() {
  const button = document.querySelector('[data-action="refresh"]');
  if (button) button.disabled = true;
  setText('dataStatus', '최신 데이터를 갱신하는 중입니다...');
  try {
    const response = await fetch('/refresh?force=true', { method: 'POST' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || result.message || `HTTP ${response.status}`);
    await boot({ refreshed: true, refreshMessage: result.message });
  } catch (error) {
    setText('dataStatus', `최신화 실패: ${error.message}`);
  } finally {
    if (button) button.disabled = false;
  }
}

async function boot(options = {}) {
  try {
    const [summary, graphs] = await Promise.all([loadSummary(), loadGraphs()]);
    const updatedAt = formatDateTime(summary.latest?.updated_at || summary.latest?.date);
    const statusMessage = options.refreshed
      ? `최신화 완료: ${updatedAt} 기준 데이터로 갱신했습니다.`
      : null;
    renderSummary(summary, statusMessage);
    renderGraphs(graphs);
    const chart = document.getElementById('forecastChart');
    const png = document.getElementById('forecastPng');
    chart.src = cacheBust('/figures/today_based_forecast.png');
    chart.dataset.mode = 'today-only-forecast';
    png.href = chart.src;
  } catch (error) {
    setText('dataStatus', `데이터 로딩 실패: ${error.message}`);
  }
}

document.querySelector('[data-action="summary"]').addEventListener('click', () => {
  document.querySelector('.metrics')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
});
document.querySelector('[data-action="forecast"]').addEventListener('click', () => {
  document.querySelector('.forecast-layout')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
});
document.querySelector('[data-action="refresh"]').addEventListener('click', refreshData);

boot();
