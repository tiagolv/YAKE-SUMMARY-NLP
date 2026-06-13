import { useState } from 'react'
import { runAblation } from '../api'
import { FlaskConical } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid
} from 'recharts'

const MODE_COLORS = {
  full: '#6c8fff',
  keywords_only: '#4ade80',
  no_keywords: '#fbbf24',
}

const MODE_LABELS = {
  full: 'Full (YAKE + External)',
  keywords_only: 'Keywords only',
  no_keywords: 'No keywords',
}

function toNumber(val) {
  if (val == null) return 0
  if (typeof val === 'number') return val
  if (typeof val === 'object') {
    if (val.coverage != null) return val.coverage
    const n = Object.values(val).find(v => typeof v === 'number')
    return n ?? 0
  }
  return parseFloat(val) || 0
}

function SummaryCard({ mode, result }) {
  const color = MODE_COLORS[mode]
  const err = result?.error
  const compression = toNumber(result?.metrics?.compression_ratio)
  const coverage = toNumber(result?.metrics?.keyword_coverage)
  return (
    <div className="surface-raised" style={{ padding: 14, flex: 1, minWidth: 0 }}>
      <div style={{
        display: 'inline-block', padding: '2px 8px', borderRadius: 4,
        background: `${color}22`, color, fontSize: 11, fontWeight: 600,
        marginBottom: 10, letterSpacing: '0.04em',
      }}>
        {MODE_LABELS[mode]}
      </div>
      {err ? (
        <div style={{ fontSize: 12, color: 'var(--red)' }}>{String(err)}</div>
      ) : (
        <>
          <p style={{ fontSize: 13, lineHeight: 1.7, color: 'var(--text-primary)', margin: '0 0 12px' }}>
            {result?.summary ?? '—'}
          </p>
          {result?.metrics && (
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Compression</div>
                <div className="mono" style={{ fontSize: 14, color }}>{compression.toFixed(3)}</div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>KW coverage</div>
                <div className="mono" style={{ fontSize: 14, color }}>{coverage.toFixed(3)}</div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function AblationComparison({ text, keywords, topK, ngramSize, temperature, savedResult, onResultSave }) {
  const [results, setResults] = useState(savedResult || null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [model, setModel] = useState('mistral')

  const run = async () => {
    if (!text?.trim()) return
    setLoading(true)
    setError(null)
    
    // Ensure keywords list structure matching backend
    const extKeywordsList = (keywords ?? '')
      .split(',')
      .map(k => k.trim())
      .filter(Boolean)

    try {
      const data = await runAblation({
        text,
        external_keywords: extKeywordsList,
        top_k: topK ?? 10,
        max_ngram_size: ngramSize ?? 3,
        temperature: temperature ?? 0.2,
        model,
      })
      
      const mappedResults = {}
      if (data?.modes) {
        data.modes.forEach(m => { mappedResults[m.mode] = m })
      }
      setResults(mappedResults)
      if (onResultSave) onResultSave(mappedResults)
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Ablation failed')
    } finally {
      setLoading(false)
    }
  }

  const chartData = results
    ? [
        {
          metric: 'Compression',
          full: toNumber(results.full?.metrics?.compression_ratio),
          yake_only: toNumber(results.yake_only?.metrics?.compression_ratio),
          no_keywords: toNumber(results.no_keywords?.metrics?.compression_ratio),
        },
        {
          metric: 'KW Coverage',
          full: toNumber(results.full?.metrics?.keyword_coverage),
          yake_only: toNumber(results.yake_only?.metrics?.keyword_coverage),
          no_keywords: toNumber(results.no_keywords?.metrics?.keyword_coverage),
        },
      ]
    : []

  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
            <FlaskConical size={14} color="var(--accent)" /> Ablation analysis
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Exclui seletivamente partes do pipeline (Keywords Externas e/ou Keywords YAKE)<br/> 
            para perceber qual é a contribuição de cada uma na qualidade do Summary final.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select value={model} onChange={e => setModel(e.target.value)} style={{ width: 'auto', minWidth: 130 }}>
            <option value="mistral">mistral</option>
            <option value="llama3">llama3</option>
            <option value="qwen2.5:7b">qwen2.5:7b</option>
          </select>
          <button
            className="btn-primary"
            onClick={run}
            disabled={loading || !text?.trim()}
          >
            {loading ? <><span className="spinner" />Running…</> : '▶  Run ablation'}
          </button>
        </div>
      </div>

      {!text?.trim() && (
        <div style={{ padding: 16, background: 'var(--bg-raised)', borderRadius: 8, fontSize: 13, color: 'var(--text-muted)', textAlign: 'center' }}>
          Enter document text in the input panel first.
        </div>
      )}

      {error && (
        <div style={{ padding: 12, background: 'var(--red-dim)', borderRadius: 8, fontSize: 13, color: 'var(--red)' }}>
          {String(error)}
        </div>
      )}

      {loading && (
        <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
          <span className="spinner" style={{ marginRight: 10 }} />
          Running 3 modes in parallel…
        </div>
      )}

      {results && !loading && (
        <>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {Object.keys(MODE_LABELS).map(mode => (
              <SummaryCard key={mode} mode={mode} result={results[mode]} />
            ))}
          </div>

          {chartData.length > 0 && (
            <div className="surface-raised" style={{ padding: 16 }}>
              <div className="label" style={{ marginBottom: 12 }}>Metric comparison</div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: -10 }}>
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} domain={[0, 1]} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-raised)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
                    formatter={(v, name) => [v.toFixed(3), MODE_LABELS[name]]}
                  />
                  {Object.entries(MODE_COLORS).map(([mode, color]) => (
                    <Bar key={mode} dataKey={mode} fill={color} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', gap: 16, marginTop: 10, flexWrap: 'wrap' }}>
                {Object.entries(MODE_LABELS).map(([mode, label]) => (
                  <div key={mode} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-secondary)' }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: MODE_COLORS[mode] }} />
                    {label}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}