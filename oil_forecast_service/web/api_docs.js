const endpoints = [
  {
    method: 'GET',
    path: '/summary',
    title: '오늘 유가 요약',
    desc: '국내 유가, WTI, Brent, 환율, 뉴스 리스크를 한 번에 확인합니다.',
    output: 'latest, forecast, news, meta, sources',
    note: '페이지 진입 시 최신 산출물이 오래됐으면 자동 갱신을 시도합니다.'
  },
  {
    method: 'GET',
    path: '/forecast',
    title: '7일 예측',
    desc: '오늘 날짜 기준 향후 7일 예측 유가와 변화 이유를 확인합니다.',
    output: 'date, predicted_domestic_price, reason',
    note: 'reason은 실제 뉴스 제목, 환율, 국제유가 흐름을 함께 설명합니다.'
  },
  {
    method: 'GET',
    path: '/graphs/list',
    title: '그래프 목록',
    desc: '웹에서 볼 수 있는 전체 그래프 파일과 원본 PNG 주소를 확인합니다.',
    output: 'filename, title, url, detail_url',
    note: '그래프 순서는 보고서 발표 흐름에 맞춰 정렬되어 있습니다.'
  },
  {
    method: 'POST',
    path: '/refresh',
    title: '최신 데이터 갱신',
    desc: '온라인 데이터를 다시 수집하고 EDA, 예측, 그래프 생성을 실행합니다.',
    output: 'status, message',
    note: '네트워크와 모델 예측이 포함되어 시간이 걸릴 수 있습니다.',
    confirm: '최신 데이터 수집과 그래프 재생성을 실행할까요? 시간이 걸릴 수 있습니다.'
  }
];

let selectedIndex = 0;

function renderEndpointList() {
  const list = document.getElementById('endpointList');
  list.innerHTML = endpoints.map((endpoint, index) => `
    <button class="endpoint ${index === selectedIndex ? 'active' : ''}" data-index="${index}">
      <span class="method ${endpoint.method === 'POST' ? 'post' : ''}">${endpoint.method} ${endpoint.path}</span>
      <small>${endpoint.title}</small>
    </button>
  `).join('');
  list.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => selectEndpoint(Number(button.dataset.index)));
  });
}

function selectEndpoint(index) {
  selectedIndex = index;
  const endpoint = endpoints[selectedIndex];
  renderEndpointList();
  document.getElementById('endpointTitle').textContent = endpoint.title;
  document.getElementById('endpointDesc').textContent = endpoint.desc;
  document.getElementById('endpointMethod').textContent = endpoint.method;
  document.getElementById('endpointPath').textContent = endpoint.path;
  document.getElementById('endpointOutput').textContent = endpoint.output;
  document.getElementById('endpointNote').textContent = endpoint.note;
  document.getElementById('openButton').href = endpoint.method === 'GET' ? endpoint.path : '#';
  document.getElementById('openButton').style.display = endpoint.method === 'GET' ? 'inline-flex' : 'none';
}

function setOutput(title, status, data) {
  document.getElementById('statusPill').textContent = status;
  document.getElementById('resultTitle').textContent = title;
  document.getElementById('resultStatus').textContent = status;
  document.getElementById('output').textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
}

async function runSelected() {
  const endpoint = endpoints[selectedIndex];
  if (endpoint.confirm && !confirm(endpoint.confirm)) return;
  setOutput(`${endpoint.method} ${endpoint.path}`, '실행 중...', '');
  try {
    const response = await fetch(endpoint.path, { method: endpoint.method });
    const data = await response.json();
    setOutput(`${endpoint.method} ${endpoint.path}`, `${response.status} ${response.statusText}`, data);
  } catch (error) {
    setOutput(`${endpoint.method} ${endpoint.path}`, '오류', String(error));
  }
}

document.getElementById('runButton').addEventListener('click', runSelected);
renderEndpointList();
selectEndpoint(0);
