import { useState } from 'react'
import { runJudge } from '../api'
import { Scale, ChevronDown, ChevronUp } from 'lucide-react'

const DIMENSIONS = [
  { key: 'fidelity', label: 'Fidelity', desc: 'No hallucinations vs. source' },
  { key: 'coverage', label: 'Coverage', desc: 'Key concepts represented' },
  { key: 'coherence', label: 'Coherence', desc: 'Fluency and consistency' },
  { key: 'keyword_relevance', label: 'Keyword relevance', desc: 'Keywords match document' },
]

const SCORE_COLORS = ['#f87171', '#fbbf24', '#fbbf24', '#4ade80', '#4ade80']

function safeScore(val) {
  if (val == null) return 0
  if (typeof val === 'number') return val
  if (typeof val === 'object') {
    const n = Object.values(val).find(v => typeof v === 'number')
    return n ?? 0
  }
  return parseFloat(val) || 0
}

function safeString(val) {
  if (val == null) return ''
  if (typeof val === 'string') return val
  if (typeof val === 'object') return JSON.stringify(val)
  return String(val)
}

function ScoreGauge({ score }) {
  const s = safeScore(score)
  const pct = ((s - 1) / 4) * 100
  const color = SCORE_COLORS[Math.round(s) - 1] ?? '#6c8fff'
  return (
    <div style={{
      width: 36, height: 36, borderRadius: '50%',
      background: `conic-gradient(${color} ${pct * 3.6}deg, var(--bg-raised) 0deg)`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        width: 26, height: 26, borderRadius: '50%',
        background: 'var(--bg-surface)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 12, fontWeight: 700, color,
        fontFamily: 'JetBrains Mono, monospace',
      }}>{s || '—'}</div>
    </div>
  )
}

function DimRow({ dim, result }) {
  const [open, setOpen] = useState(false)
  const score = result?.[dim.key]?.score
  const justification = safeString(result?.[dim.key]?.justification)

  return (
    <div className="surface-raised" style={{ marginBottom: 8, overflow: 'hidden' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', background: 'none', border: 'none', cursor: 'pointer',
          padding: '10px 14px', display: 'flex', alignItems: 'center', gap: 12,
        }}
      >
        <ScoreGauge score={score} />
        <div style={{ flex: 1, textAlign: 'left' }}>
          <div style={{ fontWeight: 500, fontSize: 13, color: 'var(--text-primary)' }}>{dim.label}</div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{dim.desc}</div>
        </div>
        {justification && (open ? <ChevronUp size={13} color="var(--text-muted)" /> : <ChevronDown size={13} color="var(--text-muted)" />)}
      </button>
      {open && justification && (
        <div style={{ padding: '0 14px 12px', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          {justification}
        </div>
      )}
    </div>
  )
}

export default function JudgePanel({ document, summary, keywords, savedResult, onResultSave }) {
  const [result, setResult] = useState(savedResult || null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [judgeModel, setJudgeModel] = useState('qwen2.5:7b')

  const canRun = document && summary

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await runJudge({
        document,
        summary,
        keywords: keywords ?? [],
        judge_model: judgeModel,
      })
      setResult(data)
      if (onResultSave) onResultSave(data)
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Judge failed')
    } finally {
      setLoading(false)
    }
  }

  const overall = result?.overall != null ? result.overall.toFixed(2) : null

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <div style={{ fontWeight: 600, marginBottom: 2, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Scale size={14} color="var(--accent)" /> LLM as a Judge
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Evaluates faithfulness, coverage, coherence & keyword relevance (1–5 scale)
          </div>
        </div>
        <select value={judgeModel} onChange={e => setJudgeModel(e.target.value)} style={{ width: 'auto', minWidth: 140 }}>
          <option value="qwen2.5:7b">qwen2.5:7b</option>
          <option value="mistral">mistral</option>
        </select>
      </div>

      {!canRun && (
        <div style={{ padding: 16, background: 'var(--bg-raised)', borderRadius: 8, fontSize: 13, color: 'var(--text-muted)', textAlign: 'center' }}>
          Run a summarisation first to enable the judge.
        </div>
      )}

      {canRun && (
        <button className="btn-primary" onClick={run} disabled={loading} style={{ alignSelf: 'flex-start' }}>
          {loading ? <><span className="spinner" />Evaluating…</> : '⚖  Evaluate summary'}
        </button>
      )}

      {error && (
        <div style={{ padding: 12, background: 'var(--red-dim)', borderRadius: 8, fontSize: 13, color: 'var(--red)' }}>
          {String(error)}
        </div>
      )}

      {result && (
        <div className="fade-in">
          {overall && (
            <div style={{
              padding: '12px 16px', marginBottom: 12,
              background: 'var(--accent-dim)', borderRadius: 8,
              display: 'flex', alignItems: 'center', gap: 12,
            }}>
              <span style={{ fontSize: 28, fontWeight: 700, color: 'var(--accent)', fontFamily: 'JetBrains Mono' }}>{overall}</span>
              <div>
                <div style={{ fontWeight: 600, fontSize: 13 }}>Overall score</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Average across {DIMENSIONS.length} dimensions</div>
              </div>
            </div>
          )}
          {DIMENSIONS.map(dim => (
            <DimRow key={dim.key} dim={dim} result={result} />
          ))}
        </div>
      )}
    </div>
  )
}