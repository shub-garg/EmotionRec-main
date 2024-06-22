import React, { useState, useEffect } from "react";
import { Box, Button, Slider, Select, MenuItem, FormControl, InputLabel } from "@mui/material";

const Settings = ({ onPlay, onPause, onStop, onVoiceChange, onPitchChange, onRateChange, onVolumeChange }) => {
  const defaultPitch = 1;
  const defaultRate = 1;
  const defaultVolume = 1;
  const defaultVoice = "Default Voice"; // Update this with your desired default voice name

  const [pitch, setPitch] = useState(defaultPitch);
  const [rate, setRate] = useState(defaultRate);
  const [volume, setVolume] = useState(defaultVolume);
  const [voice, setVoice] = useState(defaultVoice);
  const [availableVoices, setAvailableVoices] = useState([]);

  useEffect(() => {
    // Get the available voices when the component mounts
    const synth = window.speechSynthesis;
    const voices = synth.getVoices();
    setAvailableVoices(voices);

    // Set the first available voice as the default voice
    const defaultVoice = voices[0]?.name || "Default Voice";
    setVoice(defaultVoice);
  }, []);

  const handlePlay = () => {
    onPlay();
  };

  const handlePause = () => {
    onPause();
  };

  const handleStop = () => {
    onStop();
  };

  const handleVoiceChange = (event) => {
    const selectedVoice = event.target.value;
    setVoice(selectedVoice);
    onVoiceChange(selectedVoice);
  };

  const handlePitchChange = (event, newValue) => {
    setPitch(newValue);
    onPitchChange(newValue);
  };

  const handleRateChange = (event, newValue) => {
    setRate(newValue);
    onRateChange(newValue);
  };

  const handleVolumeChange = (event, newValue) => {
    setVolume(newValue);
    onVolumeChange(newValue);
  };

  return (
    <Box>
        <FormControl fullWidth>
            <InputLabel>Voice</InputLabel>
            <Select value={voice} onChange={handleVoiceChange}>
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
          onChange={handlePitchChange} // Corrected the event handling here
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

      <br />


    </Box>
  );
};

export default Settings;
