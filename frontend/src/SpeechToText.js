import { useWhisper } from '@chengsokdara/use-whisper'

const SpeechToText = () => {
  const {
    recording,
    speaking,
    transcript,
    transcripting,
    pauseRecording,
    startRecording,
    stopRecording,
  } = useWhisper({
    //apiKey: process.env.OPENAI_API_TOKEN,
    apiKey: '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'
  })

  return (
    <div>
      <p>Recording: {recording}</p>
      <p>Speaking: {speaking}</p>
      <p>Transcripting: {transcripting}</p>
      <p>Transcribed Text: {transcript.text}</p>
      <button onClick={() => startRecording()}>Start</button>
      <button onClick={() => pauseRecording()}>Pause</button>
      <button onClick={() => stopRecording()}>Stop</button>
    </div>
  )
}

export default SpeechToText;