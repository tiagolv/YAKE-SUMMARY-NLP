import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getStatus = () => api.get('/status').then(r => r.data)

export const getSamples = () => api.get('/samples').then(r => r.data)

export const getSample = (name) => api.get(`/samples/${name}`).then(r => r.data)

export const summarize = (payload) =>
  api.post('/summarize', payload).then(r => r.data)

export const runAblation = (payload) =>
  api.post('/summarize/ablation', payload).then(r => r.data)

export const runJudge = (payload) =>
  api.post('/judge', payload).then(r => r.data)
