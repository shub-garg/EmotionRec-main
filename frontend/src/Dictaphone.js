import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import MicIcon from '@mui/icons-material/Mic';
import {
  Button,
  Tooltip,
} from '@mui/material';import MicOffIcon from '@mui/icons-material/MicOff';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import './ChatPage.css';

const Dictaphone = () => {

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '10px' }}>
      <Tooltip title="Clear Speech to Text">
        <Button
          sx={{
            backgroundColor: '#233036',
            '&:hover': {
              backgroundColor: '#447796',
            },
          }}
          variant="contained"
        >
          Clear STT
        </Button>
      </Tooltip>
    </div>
  );
};

export default Dictaphone;
