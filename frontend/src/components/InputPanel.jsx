import { useState, useEffect } from 'react'
import { getSamples, getSample } from '../api'
import { FileText, ChevronDown, ChevronUp } from 'lucide-react'

export default function InputPanel({ onSubmit, loading }) {
  const [text, setText] = useState('')
  const [keywords, setKeywords] = useState('')
  const [topK, setTopK] = useState(10)
  const [ngramSize, setNgramSize] = useState(3)
  const [temperature, setTemperature] = useState(0.2)
  const [mode, setMode] = useState('full')
  const [model, setModel] = useState('mistral')
  const [advOpen, setAdvOpen] = useState(false)
  const [samples, setSamples] = useState([])
  const [selectedSample, setSelectedSample] = useState('')

  useEffect(() => {
    getSamples()
      .then(d => setSamples(d.samples ?? []))
      .catch(() => {})
  }, [])

  const loadSample = async (name) => {
    if (!name) return
    setSelectedSample(name)
    try {
      const d = await getSample(name)
      setText(d.text ?? '')
      if (d.keywords?.length) setKeywords(d.keywords.join(', '))
    } catch {}
  }

  const handleSubmit = () => {
    if (!text.trim()) return
    const external_keywords = keywords
      .split(',')
      .map(k => k.trim())
      .filter(Boolean)

    onSubmit({
      text,
      external_keywords,
      top_k: topK,
      ablation_mode: mode,
      model,
      max_ngram_size: ngramSize,
      temperature
    })
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Sample picker */}
      {samples.length > 0 && (
        <div>
          <div className="label" style={{ marginBottom: 6 }}>Load example</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {samples.map(s => (
              <button
                key={s}
                className="btn-secondary"
                style={{
                  fontSize: 11,
                  padding: '4px 10px',
                  borderColor: selectedSample === s ? 'var(--accent)' : undefined,
                  color: selectedSample === s ? 'var(--accent)' : undefined,
                }}
                onClick={() => loadSample(s)}
              >
                {s.replace(/_/g, ' ').replace('abstract ', '')}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Text input */}
      <div>
        <div className="label" style={{ marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
          <FileText size={11} /> Document
        </div>
        <textarea
          rows={10}
          placeholder="Paste your abstract or document text here…"
          value={text}
          onChange={e => setText(e.target.value)}
          style={{ minHeight: 180 }}
        />
        {text && (
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
            {text.split(/\s+/).filter(Boolean).length} words
          </div>
        )}
      </div>

      {/* Keywords */}
      <div>
        <div className="label" style={{ marginBottom: 6 }}>External keywords <span style={{ fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>(optional, comma-separated)</span></div>
        <input
          type="text"
          placeholder="e.g. federated learning, privacy, edge devices"
          value={keywords}
          onChange={e => setKeywords(e.target.value)}
        />
      </div>

      {/* Top-K slider */}
      <div>
        <div className="label" style={{ marginBottom: 6, display: 'flex', justifyContent: 'space-between' }}>
          <span>YAKE Top-K</span>
          <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{topK}</span>
        </div>
        <input
          type="range"
          min={3} max={30} step={1}
          value={topK}
          onChange={e => setTopK(Number(e.target.value))}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
          <span>3</span><span>30</span>
        </div>
      </div>

      {/* Advanced accordion */}
      <div className="surface-raised" style={{ overflow: 'hidden' }}>
        <button
          onClick={() => setAdvOpen(o => !o)}
          style={{
            width: '100%', background: 'none', border: 'none', cursor: 'pointer',
            padding: '10px 14px', display: 'flex', justifyContent: 'space-between',
            alignItems: 'center', color: 'var(--text-secondary)', fontSize: 13,
          }}
        >
          <span>Advanced options</span>
          {advOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {advOpen && (
          <div style={{ padding: '0 14px 14px', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div>
              <div className="label" style={{ marginBottom: 6 }}>Mode</div>
              <select value={mode} onChange={e => setMode(e.target.value)}>
                <option value="full">Full (YAKE + External)</option>
                <option value="yake_only">YAKE only</option>
                <option value="external_only">External only</option>
                <option value="no_keywords">No keywords</option>
              </select>
            </div>
            <div>
              <div className="label" style={{ marginBottom: 6 }}>Model</div>
              <select value={model} onChange={e => setModel(e.target.value)}>
                <option value="mistral">mistral</option>
                <option value="llama3">llama3</option>
                <option value="qwen2.5:7b">qwen2.5:7b</option>
                <option value="llama3.2">llama3.2</option>
              </select>
            </div>
            
            <div>
              <div className="label" style={{ marginBottom: 6, display: 'flex', justifyContent: 'space-between' }}>
                <span>N-Gram Size</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{ngramSize}</span>
              </div>
              <input
                type="range"
                min={1} max={4} step={1}
                value={ngramSize}
                onChange={e => setNgramSize(Number(e.target.value))}
              />
            </div>

            <div>
              <div className="label" style={{ marginBottom: 6, display: 'flex', justifyContent: 'space-between' }}>
                <span>Temperature</span>
                <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{temperature}</span>
              </div>
              <input
                type="range"
                min={0} max={1} step={0.1}
                value={temperature}
                onChange={e => setTemperature(Number(e.target.value))}
              />
            </div>
          </div>
        )}
      </div>

      {/* Submit */}
      <button
        className="btn-primary"
        onClick={handleSubmit}
        disabled={loading || !text.trim()}
        style={{ width: '100%', justifyContent: 'center', padding: '12px' }}
      >
        {loading ? <><span className="spinner" />Running…</> : '▶  Run summarisation'}
      </button>
    </div>
  )
}
