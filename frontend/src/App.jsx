import { useState } from 'react'
import { summarize } from './api'
import StatusBanner from './components/StatusBanner'
import InputPanel from './components/InputPanel'
import MetricsCard from './components/MetricsCard'
import JudgePanel from './components/JudgePanel'
import AblationComparison from './components/AblationComparison'
import { BookOpen, BarChart2, Scale, FlaskConical, Copy, Check } from 'lucide-react'

const TABS = [
  { id: 'summary', label: 'Summary', icon: BookOpen },
  { id: 'metrics', label: 'Metrics', icon: BarChart2 },
  { id: 'judge', label: 'Judge', icon: Scale },
  { id: 'ablation', label: 'Ablation', icon: FlaskConical },
]

// keywords may be [[kw, score], ...] or {kw: score}
function extractKeywordNames(keywordsObj) {
  if (!keywordsObj) return []
  if (Array.isArray(keywordsObj.keywords)) {
    return keywordsObj.keywords.map(k => k.keyword)
  }
  return []
}

function normaliseKeywords(keywordsObj) {
  if (!keywordsObj) return []
  if (Array.isArray(keywordsObj.keywords)) {
    return keywordsObj.keywords.map(k => [k.keyword, k.score])
  }
  return []
}

function KeywordPill({ kw, score }) {
  const opacity = Math.max(0.4, 1 - score * 2)
  return (
    <span style={{
      display: 'inline-block', padding: '3px 9px', borderRadius: 20,
      background: `rgba(108,143,255,${opacity * 0.25})`,
      border: `1px solid rgba(108,143,255,${opacity * 0.5})`,
      color: 'var(--text-secondary)', fontSize: 12, margin: '2px 3px',
    }}>
      {kw}
    </span>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }
  return (
    <button onClick={copy} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
      {copied ? <Check size={13} color="var(--green)" /> : <Copy size={13} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('summary')
  const [runs, setRuns] = useState([])
  const [activeRunId, setActiveRunId] = useState(null)
  const [loading, setLoading] = useState(false)

  const activeRun = runs.find(r => r.id === activeRunId)
  const result = activeRun?.result
  const lastParams = activeRun?.params
  const error = activeRun?.error

  const updateActiveRun = (updates) => {
    if (!activeRunId) return
    setRuns(prev => prev.map(r => r.id === activeRunId ? { ...r, ...updates } : r))
  }

  const handleSubmit = async (params) => {
    const newRunId = Date.now()
    const newRun = {
      id: newRunId,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      params,
      result: null,
      error: null,
    }
    
    setRuns(prev => [newRun, ...prev])
    setActiveRunId(newRunId)
    setLoading(true)

    try {
      const data = await summarize(params)
      setRuns(prev => prev.map(r => r.id === newRunId ? { ...r, result: data } : r))
      setActiveTab('summary')
    } catch (e) {
      setRuns(prev => prev.map(r => r.id === newRunId ? { ...r, error: e.response?.data?.detail ?? e.message ?? 'Unknown error' } : r))
    } finally {
      setLoading(false)
    }
  }

  const kwPairs = normaliseKeywords(result?.keywords)

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        padding: '0 24px',
        height: 52,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-surface)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6,
            background: 'linear-gradient(135deg, #6c8fff, #a78bfa)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, fontWeight: 700, color: '#fff', fontFamily: 'Sora',
          }}>C</div>
          <span style={{ fontFamily: 'Sora', fontWeight: 700, fontSize: 16, letterSpacing: '-0.02em' }}>CLNP</span>
          <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>/ Keyword-Guided Summarisation</span>
        </div>
        <div style={{ width: 280 }}>
          <StatusBanner />
        </div>
      </header>

      {/* Body */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Sidebar */}
        <aside style={{
          width: 320,
          borderRight: '1px solid var(--border)',
          padding: 20,
          overflowY: 'auto',
          background: 'var(--bg-surface)',
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
        }}>
          <InputPanel onSubmit={handleSubmit} loading={loading} />

          {runs.length > 0 && (
            <div style={{ paddingTop: 20, borderTop: '1px solid var(--border)' }}>
              <div className="label" style={{ marginBottom: 10 }}>History ({runs.length})</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {runs.map(r => (
                  <button
                    key={r.id}
                    onClick={() => setActiveRunId(r.id)}
                    style={{
                      textAlign: 'left', padding: '10px 12px', borderRadius: 8, fontSize: 12,
                      background: r.id === activeRunId ? 'var(--accent-dim)' : 'transparent',
                      border: `1px solid ${r.id === activeRunId ? 'var(--accent)' : 'var(--border)'}`,
                      color: r.id === activeRunId ? 'var(--accent)' : 'var(--text-secondary)',
                      cursor: 'pointer', display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center'
                    }}
                  >
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 180 }}>
                      {r.params.text.slice(0, 30)}…
                    </span>
                    <span style={{ fontSize: 10, opacity: 0.7 }}>{r.time}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* Main panel */}
        <main style={{ flex: 1, overflowY: 'auto', padding: 24 }}>
          {/* Tab bar */}
          <div className="tab-bar">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`tab ${activeTab === id ? 'active' : ''}`}
                onClick={() => setActiveTab(id)}
                style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              >
                <Icon size={13} /> {label}
              </button>
            ))}
          </div>

          {error && (
            <div style={{ padding: 14, background: 'var(--red-dim)', borderRadius: 8, color: 'var(--red)', marginBottom: 16, fontSize: 13 }}>
              {error}
            </div>
          )}

          {loading && (
            <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
              <span className="spinner" style={{ marginBottom: 12, display: 'inline-block' }} />
              <div style={{ fontSize: 13 }}>Generating summary…</div>
            </div>
          )}

          {!loading && !result && !error && activeTab !== 'ablation' && (
            <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>◎</div>
              <div style={{ fontWeight: 600, marginBottom: 6, color: 'var(--text-secondary)' }}>No results yet</div>
              <div style={{ fontSize: 13 }}>Enter a document in the sidebar and click Run.</div>
            </div>
          )}

          {/* Summary tab */}
          <div style={{ display: !loading && result && activeTab === 'summary' ? 'block' : 'none' }}>
            <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div className="surface-raised" style={{ padding: 18 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div className="label">Summary</div>
                  {result?.summary && <CopyButton text={result.summary} />}
                </div>
                <p style={{ margin: 0, lineHeight: 1.8, fontSize: 14 }}>
                  {result?.summary ?? result?.degraded_summary ?? 'No summary returned.'}
                </p>
                {result?.degraded_summary && (
                  <div style={{ marginTop: 10, padding: '8px 12px', background: 'var(--amber-dim)', borderRadius: 6, fontSize: 12, color: 'var(--amber)' }}>
                    ⚠ LLM unavailable — showing extractive fallback
                  </div>
                )}
              </div>

              {kwPairs.length > 0 && (
                <div className="surface-raised" style={{ padding: 16 }}>
                  <div className="label" style={{ marginBottom: 10 }}>Extracted keywords (YAKE)</div>
                  <div>
                    {kwPairs.map(([kw, score], i) => (
                      <KeywordPill key={i} kw={kw} score={score} />
                    ))}
                  </div>
                </div>
              )}

              {result?.prompt_used && (
                <details style={{ cursor: 'pointer' }}>
                  <summary style={{ fontSize: 12, color: 'var(--text-muted)', userSelect: 'none', listStyle: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span>▶</span> View prompt sent to LLM
                  </summary>
                  <pre style={{
                    marginTop: 8, padding: 14,
                    background: 'var(--bg-raised)', borderRadius: 8,
                    fontSize: 11, lineHeight: 1.6, whiteSpace: 'pre-wrap',
                    color: 'var(--text-secondary)', fontFamily: 'JetBrains Mono',
                    border: '1px solid var(--border)',
                  }}>
                    {result.prompt_used}
                  </pre>
                </details>
              )}
            </div>
          </div>

          {/* Metrics tab */}
          <div style={{ display: !loading && result && activeTab === 'metrics' ? 'block' : 'none' }}>
            <MetricsCard metrics={result?.metrics} keywords={result?.keywords} />
          </div>

          {/* Judge tab */}
          <div style={{ display: activeTab === 'judge' ? 'block' : 'none' }}>
            {activeRunId && <JudgePanel
              key={activeRunId}
              document={lastParams?.text}
              summary={result?.summary}
              keywords={extractKeywordNames(result?.keywords)}
              savedResult={activeRun?.judgeResult}
              onResultSave={(data) => updateActiveRun({ judgeResult: data })}
            />}
          </div>

          {/* Ablation tab */}
          <div style={{ display: activeTab === 'ablation' ? 'block' : 'none' }}>
            {activeRunId && <AblationComparison
              key={activeRunId}
              text={lastParams?.text}
              keywords={lastParams?.external_keywords ? lastParams.external_keywords.join(', ') : ''}
              topK={lastParams?.top_k}
              ngramSize={lastParams?.max_ngram_size}
              temperature={lastParams?.temperature}
              savedResult={activeRun?.ablationResult}
              onResultSave={(data) => updateActiveRun({ ablationResult: data })}
            />}
          </div>
        </main>
      </div>
    </div>
  )
}