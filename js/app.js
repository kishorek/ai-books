const BOOKS = {
  'agentic_ai_book': {
    title: 'Agentic AI',
    chapters: [
      { file: '00-introduction.md', title: 'Introduction' },
      { file: '01-foundations.md', title: 'Ch 1: Agentic AI Fundamentals' },
      { file: '02-agent-architectures.md', title: 'Ch 2: Agent Architectures' },
      { file: '03-multi-agent-design.md', title: 'Ch 3: Multi-Agent Design' },
      { file: '04-state-management.md', title: 'Ch 4: State Management' },
      { file: '05-context-engineering.md', title: 'Ch 5: Context Engineering' },
      { file: '06-deterministic-systems.md', title: 'Ch 6: Deterministic Systems' },
      { file: '07-reliability-engineering.md', title: 'Ch 7: Reliability Engineering' },
      { file: '08-distributed-systems.md', title: 'Ch 8: Distributed Systems' },
      { file: '09-workflow-orchestration.md', title: 'Ch 9: Workflow Orchestration' },
      { file: '10-scalability.md', title: 'Ch 10: Scalability' },
      { file: '11-governance.md', title: 'Ch 11: Enterprise Governance' },
      { file: '12-llmops-agentops.md', title: 'Ch 12: LLMOps & AgentOps' },
      { file: '13-advanced-patterns.md', title: 'Ch 13: Advanced Patterns' },
      { file: '14-learning-roadmap.md', title: 'Ch 14: Learning Roadmap' },
    ]
  },
  'GenAI Applications': {
    title: 'GenAI Applications',
    chapters: [
      { file: '00-introduction.md', title: 'Introduction' },
      { file: '01-genai-foundations.md', title: 'Ch 1: GenAI Foundations' },
      { file: '02-llm-fundamentals.md', title: 'Ch 2: LLM Fundamentals' },
      { file: '03-model-architectures.md', title: 'Ch 3: Model Architectures' },
      { file: '04-prompt-engineering.md', title: 'Ch 4: Prompt Engineering' },
      { file: '05-llm-apis.md', title: 'Ch 5: LLM APIs' },
      { file: '06-building-apps.md', title: 'Ch 6: Building AI Apps' },
      { file: '07-context-engineering.md', title: 'Ch 7: Context Engineering' },
      { file: '08-rag.md', title: 'Ch 8: RAG' },
      { file: '09-tool-calling.md', title: 'Ch 9: Tool Calling' },
      { file: '10-ai-agents.md', title: 'Ch 10: AI Agents' },
      { file: '11-agent-frameworks.md', title: 'Ch 11: Agent Frameworks' },
      { file: '12-memory-systems.md', title: 'Ch 12: Memory Systems' },
      { file: '13-structured-outputs.md', title: 'Ch 13: Structured Outputs' },
      { file: '14-evaluation.md', title: 'Ch 14: Evaluation' },
      { file: '15-llmops.md', title: 'Ch 15: LLMOps' },
      { file: '16-security.md', title: 'Ch 16: Security' },
      { file: '17-enterprise-architecture.md', title: 'Ch 17: Enterprise Architecture' },
      { file: '18-multimodal-ai.md', title: 'Ch 18: Multimodal AI' },
      { file: '19-production-products.md', title: 'Ch 19: Production Products' },
      { file: '20-future-architectures.md', title: 'Ch 20: Future Architectures' },
    ]
  },
  'RAG': {
    title: 'RAG',
    chapters: [
      { file: '00-introduction.md', title: 'Introduction' },
      { file: '01-rag-fundamentals.md', title: 'Ch 1: RAG Fundamentals' },
      { file: '02-information-retrieval.md', title: 'Ch 2: Information Retrieval' },
      { file: '03-document-processing.md', title: 'Ch 3: Document Processing' },
      { file: '04-chunking-strategies.md', title: 'Ch 4: Chunking Strategies' },
      { file: '05-embeddings.md', title: 'Ch 5: Embeddings' },
      { file: '06-vector-databases.md', title: 'Ch 6: Vector Databases' },
      { file: '07-retrieval-engineering.md', title: 'Ch 7: Retrieval Engineering' },
      { file: '08-reranking.md', title: 'Ch 8: Re-ranking' },
      { file: '09-context-engineering-rag.md', title: 'Ch 9: Context Engineering' },
      { file: '10-knowledge-graph-rag.md', title: 'Ch 10: Knowledge Graph RAG' },
      { file: '11-multimodal-rag.md', title: 'Ch 11: Multimodal RAG' },
      { file: '12-agentic-rag.md', title: 'Ch 12: Agentic RAG' },
      { file: '13-rag-evaluation.md', title: 'Ch 13: RAG Evaluation' },
      { file: '14-production-architecture.md', title: 'Ch 14: Production Architecture' },
      { file: '15-security-governance.md', title: 'Ch 15: Security & Governance' },
      { file: '16-advanced-enterprise-rag.md', title: 'Ch 16: Advanced Enterprise RAG' },
    ]
  }
};

// DOM refs
const sidebar = document.getElementById('sidebar');
const sidebarNav = document.getElementById('sidebar-nav');
const sidebarTitle = document.getElementById('sidebar-book-title');
const sidebarClose = document.getElementById('sidebar-close');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const menuBtn = document.getElementById('menu-btn');
const breadcrumb = document.getElementById('breadcrumb');
const article = document.getElementById('article');
const loading = document.getElementById('loading');
const themeToggle = document.getElementById('theme-toggle');

let currentBook = null;
let currentChapter = null;
let mermaidCounter = 0;

// Mermaid setup
function initMermaid() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  mermaid.initialize({
    startOnLoad: false,
    theme: isDark ? 'dark' : 'default',
    securityLevel: 'loose',
    fontFamily: 'Inter, sans-serif',
    flowchart: { useMaxWidth: true, htmlLabels: true, curve: 'basis' },
    sequence: { useMaxWidth: true, diagramMarginX: 20, diagramMarginY: 20 },
    gantt: { useMaxWidth: true },
  });
}

// Theme
function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
}

themeToggle?.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  initMermaid();
  // Re-render mermaid in current article
  if (currentBook && currentChapter) {
    renderMermaidDiagrams();
  }
});

// Sidebar
function openSidebar() {
  sidebar.classList.add('open');
  sidebarOverlay.classList.add('active');
}

function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
}

menuBtn?.addEventListener('click', openSidebar);
sidebarClose?.addEventListener('click', closeSidebar);
sidebarOverlay?.addEventListener('click', closeSidebar);

// Navigation
function buildSidebar(bookKey) {
  const book = BOOKS[bookKey];
  if (!book) return;

  currentBook = bookKey;
  sidebarTitle.textContent = book.title;
  sidebarNav.innerHTML = '';

  book.chapters.forEach((ch, i) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = `#${bookKey}/${ch.file.replace('.md', '')}`;
    a.textContent = ch.title;
    if (i === 0) a.classList.add('chapter-intro');
    a.addEventListener('click', () => closeSidebar());
    li.appendChild(a);
    sidebarNav.appendChild(li);
  });
}

function highlightActive() {
  const links = sidebarNav.querySelectorAll('a');
  links.forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === `#${currentBook}/${currentChapter}`);
  });
}

// Render markdown
async function loadChapter(bookKey, chapterSlug) {
  const book = BOOKS[bookKey];
  if (!book) return;

  const chapter = book.chapters.find(ch => ch.file.replace('.md', '') === chapterSlug);
  if (!chapter) return;

  currentBook = bookKey;
  currentChapter = chapterSlug;

  // Show loading
  article.innerHTML = '';
  article.appendChild(loading);
  loading.style.display = '';

  // Update sidebar
  buildSidebar(bookKey);
  highlightActive();

  // Update breadcrumb
  breadcrumb.innerHTML = `<a href="index.html">Books</a> &rsaquo; <a href="#${bookKey}">${book.title}</a> &rsaquo; ${chapter.title}`;

  // Update doc title
  document.title = `${chapter.title} — ${book.title}`;

  try {
    const encodedPath = encodeURIComponent(chapter.file);
    const resp = await fetch(`${bookKey}/${encodedPath}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const md = await resp.text();

    // Parse markdown
    article.innerHTML = marked.parse(md);

    // Render mermaid
    await renderMermaidDiagrams();

    // Scroll to top
    window.scrollTo(0, 0);
  } catch (err) {
    article.innerHTML = `<div class="loading"><p>Failed to load chapter.</p><p style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.5rem;">${err.message}</p></div>`;
  }
}

async function renderMermaidDiagrams() {
  const mermaidDivs = article.querySelectorAll('.mermaid');
  if (mermaidDivs.length === 0) return;

  initMermaid();

  for (let i = 0; i < mermaidDivs.length; i++) {
    const div = mermaidDivs[i];
    const graphDefinition = div.textContent.trim();
    const id = `mermaid-${++mermaidCounter}`;

    try {
      const { svg } = await mermaid.render(id, graphDefinition);
      div.innerHTML = svg;
    } catch (err) {
      console.warn(`Mermaid render error (diagram ${i + 1}):`, err);
      div.innerHTML = `<pre style="color:red;font-size:0.8rem;text-align:left;overflow:auto;">${err.message || 'Diagram render failed'}</pre>`;
    }
  }
}

// Hash routing
function handleHash() {
  const hash = window.location.hash.slice(1);
  if (!hash) {
    window.location.hash = '#agentic_ai_book';
    return;
  }

  const parts = hash.split('/');
  const bookKey = decodeURIComponent(parts[0]);

  if (parts.length > 1) {
    const chapterSlug = parts.slice(1).join('/');
    loadChapter(bookKey, chapterSlug);
  } else {
    // Default to first chapter of the book
    const book = BOOKS[bookKey];
    if (book) {
      window.location.hash = `#${bookKey}/${book.chapters[0].file.replace('.md', '')}`;
    }
  }
}

window.addEventListener('hashchange', handleHash);

// Init
initTheme();
initMermaid();
handleHash();
