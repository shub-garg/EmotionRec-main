import React, { useState, useEffect } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from "@mui/material";

const TextToSpeech = ({ text }) => {
  const [isPaused, setIsPaused] = useState(false);
  const [utterance, setUtterance] = useState(null);
  const [voice, setVoice] = useState(null);
  const [pitch, setPitch] = useState(1);
  const [rate, setRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [availableVoices, setAvailableVoices] = useState([]);

  useEffect(() => {
    const synth = window.speechSynthesis;
    const u = new SpeechSynthesisUtterance(text);
    const voices = synth.getVoices();

    setUtterance(u);
    setVoice(voices[0]);
    setAvailableVoices(voices);

    return () => {
      synth.cancel();
    };
  }, [text]);

  const handlePlay = () => {
    const synth = window.speechSynthesis;

    if (isPaused) {
      synth.resume();
    } else {
      utterance.voice = voice;
      utterance.pitch = pitch;
      utterance.rate = rate;
      utterance.volume = volume;
      synth.speak(utterance);
    }

    setIsPaused(false);
  };

  const handlePause = () => {
    const synth = window.speechSynthesis;

    synth.pause();

    setIsPaused(true);
  };

  const handleStop = () => {
    const synth = window.speechSynthesis;

    synth.cancel();

    setIsPaused(false);
  };

  const handleVoiceChange = (event) => {
    const selectedVoiceName = event.target.value;
    const selectedVoice = availableVoices.find((v) => v.name === selectedVoiceName);
    setVoice(selectedVoice);
  };

  const handlePitchChange = (event, newValue) => {
    setPitch(newValue);
  };

  const handleRateChange = (event, newValue) => {
    setRate(newValue);
  };

  const handleVolumeChange = (event, newValue) => {
    setVolume(newValue);
  };

  const handleOpenSettings = () => {
    setIsSettingsOpen(true);
  };

  const handleCloseSettings = () => {
    setIsSettingsOpen(false);
  };

  return (
    <div>
      <Button variant="contained" onClick={handleOpenSettings}>
        Open Settings
      </Button>

      <Dialog open={isSettingsOpen} onClose={handleCloseSettings}>
        <DialogTitle>Text to Speech Settings</DialogTitle>
        <DialogContent>
          <Box>
            <FormControl fullWidth>
              <InputLabel>Voice</InputLabel>
              <Select value={voice?.name || "Default Voice"} onChange={handleVoiceChange}>
                {availableVoices.map((voice) => (
                  <MenuItem key={voice.name} value={voice.name}>
                    {voice.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <br />

            <Box sx={{ width: 200 }}>
              <InputLabel>Pitch</InputLabel>
              <Slider
                value={pitch}
                min={0.5}
                max={2}
                step={0.1}
                onChange={handlePitchChange}
                valueLabelDisplay="auto"
                valueLabelFormat={(x) => x.toFixed(1)}
              />
            </Box>

            <br />

            <Box sx={{ width: 200 }}>
              <InputLabel>Speed</InputLabel>
              <Slider
                value={rate}
                min={0.5}
                max={2}
                step={0.1}
                onChange={handleRateChange}
                valueLabelDisplay="auto"
                valueLabelFormat={(x) => x.toFixed(1)}
              />
            </Box>

            <br />

            <Box sx={{ width: 200 }}>
              <InputLabel>Volume</InputLabel>
              <Slider
                value={volume}
                min={0}
                max={1}
                step={0.1}
                onChange={handleVolumeChange}
                valueLabelDisplay="auto"
                valueLabelFormat={(x) => x.toFixed(1)}
              />
            </Box>
          </Box>
          <Button variant="contained" onClick={handlePlay}>
            Play
          </Button>
          <Button variant="contained" onClick={handlePause}>
            Pause
          </Button>
          <Button variant="contained" onClick={handleStop}>
            Stop
          </Button>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseSettings} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
      <br />

    </div>
  );
};

export default TextToSpeech;