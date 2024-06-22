import React, { useState, useRef, useEffect } from 'react';
import {
  TextField,
  Button,
  Paper,
  Tooltip,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Box,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import './ChatPage.css';
import VolumeMuteIcon from '@mui/icons-material/VolumeMute';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';import VolumeOffIcon from '@mui/icons-material/VolumeOff';
import SettingsIcon from '@mui/icons-material/Settings';
import Dictaphone from './Dictaphone';
import WebcamCompStreamCapture from './WebcamStreamCapture';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

const ChatPage = () => {
  const [message, setMessage] = useState('');
  const [aiLatestMessage, setAiLatestMessage] = useState('');
  const [dictaphoneTranscript, setDictaphoneTranscript] = useState('');
  const [chatLog, setChatLog] = useState([]);
  const videoPlayerRef = useRef(null);
  const chatLogRef = useRef(null);
  const dictaphoneRef = useRef(null);
  const synthRef = useRef(window.speechSynthesis);
  const [synthesisPlaying, setSynthesisPlaying] = useState(false);

  // Text-to-speech settings
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [selectedPitch, setSelectedPitch] = useState(1);
  const [selectedRate, setSelectedRate] = useState(1);
  const [selectedVolume, setSelectedVolume] = useState(1);
  const [availableVoices, setAvailableVoices] = useState([]);
  const [latestAiMessageIndex, setLatestAiMessageIndex] = useState(null);
  const [muteTTS, setMuteTTS] = useState(false);

  // Temp TTS
  const [selectedVoiceTemp, setSelectedVoiceTemp] = useState(null);
  const [selectedPitchTemp, setSelectedPitchTemp] = useState(1);
  const [selectedRateTemp, setSelectedRateTemp] = useState(1);
  const [selectedVolumeTemp, setSelectedVolumeTemp] = useState(1);
  const [initialVolume, setInitialVolume] = useState(selectedVolume);

  // Settings dialog state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Text-to-speech status
  const [textToSpeechStatus, setTextToSpeechStatus] = useState({});

  //Speech to text 
  const [isListening, setIsListening] = useState(false);
  const { transcript, resetTranscript, browserSupportsSpeechRecognition } = useSpeechRecognition();

  // Video playback state
  const [videoState, setVideoState] = useState('idle');
  const [isPlaying, setIsPlaying] = useState(false);

  // Add state variables for selected avatar and video source
  const [selectedAvatar, setSelectedAvatar] = useState('man');
  const [videoSource, setVideoSource] = useState('ai-man.webm');
  const [previousVideoSource, setPreviousVideoSource] = useState('ai-man.webm'); // Store the previous video source

  // Functions for handling settings changes
  const handleVoiceChange = (event) => setSelectedVoiceTemp(findVoiceByName(event.target.value));
  const handlePitchChange = (event, newValue) => setSelectedPitchTemp(newValue);
  const handleRateChange = (event, newValue) => setSelectedRateTemp(newValue);
  const handleVolumeChange = (event, newValue) => setSelectedVolumeTemp(newValue);

  // Functions for opening and closing settings dialog
  const handleOpenSettings = () => setIsSettingsOpen(true);
  
  const handleCloseSettings = (saveSettings) => {
    if (saveSettings) {
      setSelectedVoice(selectedVoiceTemp);
      setSelectedPitch(selectedPitchTemp);
      setSelectedRate(selectedRateTemp);
      setSelectedVolume(selectedVolumeTemp);
      // Update the video source if it has changed in the settings
      if (videoSource !== previousVideoSource) {
        setPreviousVideoSource(videoSource); // Save the current video source
        if (videoPlayerRef.current) {
          videoPlayerRef.current.src = process.env.PUBLIC_URL + '/' + videoSource;
          videoPlayerRef.current.load(); // Load the new video source
        }
        playTextToSpeech(aiLatestMessage, selectedVoiceTemp, selectedPitchTemp, selectedRateTemp, selectedVolumeTemp, true);
      }
    } else {
      setSelectedVoiceTemp(selectedVoice);
      setSelectedPitchTemp(selectedPitch);
      setSelectedRateTemp(selectedRate);
      setSelectedVolumeTemp(selectedVolume);
      // Reset the video source to the previous source
      setVideoSource(previousVideoSource);

      if (videoPlayerRef.current) {
        videoPlayerRef.current.src = process.env.PUBLIC_URL + '/' + previousVideoSource;
        videoPlayerRef.current.load(); // Load the new video source
      }
    }
    setIsSettingsOpen(false);
  };
  
  
  // Other utility functions
  const findVoiceByName = (voiceName) => availableVoices.find((v) => v.name === voiceName);

  
  useEffect(() => {
    const video = videoPlayerRef.current;
    video.muted = true;
    return () => {
      video.muted = false; // Unmute the video when the component unmounts
    };
  }, []);

  useEffect(() => {
    const chatLogContainer = chatLogRef.current;
    if (chatLogContainer) {
      chatLogContainer.scrollTop = chatLogContainer.scrollHeight;
    }
  }, [chatLog]);

  useEffect(() => {
    const synth = window.speechSynthesis;

    const handleVoicesChanged = () => {
      const voices = synth.getVoices();
      setAvailableVoices(voices);

      // Set the first voice as the default selectedVoice
      if (!selectedVoice) {
        setSelectedVoice(voices[0]);
        setSelectedVoiceTemp(voices[0]); // Set the default value for the temporary state variable
      }
    };

    // Listen for the 'voiceschanged' event to get the voices
    synth.addEventListener('voiceschanged', handleVoicesChanged);

    // Clean up the event listener when the component unmounts
    return () => {
      synth.removeEventListener('voiceschanged', handleVoicesChanged);
    };
  }, []);

  useEffect(() => {
    setSelectedVoiceTemp(selectedVoice);
    setSelectedPitchTemp(selectedPitch);
    setSelectedRateTemp(selectedRate);
    setSelectedVolumeTemp(selectedVolume);
  }, [selectedVoice, selectedPitch, selectedRate, selectedVolume]);

  useEffect(() => {
    const synth = window.speechSynthesis;
    const voices = synth.getVoices();
    setAvailableVoices(voices);

    // Set the first voice as the default selectedVoice
    setSelectedVoice(voices[0]);
  }, []);

  useEffect(() => {
    const video = videoPlayerRef.current;
    if (video) {
      video.currentTime = 0;
      if (videoState === 'play') {
        video.play().then(() => {
          setIsPlaying(true);
        }).catch((error) => {
          console.error('Error playing video:', error);
        });
      } else if (videoState === 'idle') {
        video.pause();
        setIsPlaying(false);
      }
    }
  }, [videoState]);

  useEffect(() => {
    if (aiLatestMessage) {
      playTextToSpeech(aiLatestMessage, selectedVoice, selectedPitch, selectedRate, selectedVolume, true);
    }
    
    // Pause the video when text-to-speech stops talking
    if (videoState === 'play' && !synthesisPlaying) {
      handlePause();
    }
  }, [aiLatestMessage, selectedVoice, selectedPitch, selectedRate, selectedVolume, synthesisPlaying]);

  useEffect(() => {
    handleTranscriptChange(transcript);
  }, [transcript]);

  const playTextToSpeech = (text, voice, pitch, rate, volume, shouldSkip) => {
    return new Promise((resolve, reject) => {
      if (synthesisPlaying) {
        resolve(); // If text-to-speech is already playing, resolve immediately
      }
  
      const synth = synthRef.current;
      const u = new SpeechSynthesisUtterance(text);
      const voices = availableVoices;
      const selectedVoiceIndex = voices.findIndex((v) => v.name === voice?.name);
      u.voice = voices[selectedVoiceIndex]; // Use the selected voice, or default to the first voice
      u.pitch = pitch || selectedPitch; // Use the selected pitch, or default to state value
      u.rate = rate || selectedRate; // Use the selected rate, or default to state value
      u.volume = volume || selectedVolume; // Use the selected volume, or default to state value
  
      if (!shouldSkip) {
        setSynthesisPlaying(true); // Mark text-to-speech playback as started
        setVideoState('play');
        synth.speak(u);
        u.onend = () => {
          setSynthesisPlaying(false); // Mark text-to-speech playback as finished
          setVideoState('idle');
          resolve(); // Resolve the promise on successful completion
        };
        u.onerror = (error) => {
          setSynthesisPlaying(false); // Mark text-to-speech playback as finished
          setVideoState('idle');
          reject(error); // Reject the promise if there's an error (interruption)
        };
      } else {
        resolve(); // Resolve immediately if skipping text-to-speech
      }
    });
  };
  
  const handlePlay = () => {
    const video = videoPlayerRef.current;
    video.play().then(() => {
      setIsPlaying(true);
    }).catch((error) => {
      console.error('Error playing video:', error);
    });
  };

  const handlePause = () => {
    const video = videoPlayerRef.current;
    video.pause();
    setIsPlaying(false);
  };

  const handleChatMessage = (message) => {
    setVideoState(message === 'play' ? 'play' : 'idle');
  };

  const handleToggleMuteTTS = () => {
    if (muteTTS) {
      synthRef.current.cancel(); // Stop ongoing TTS if muted
      setSelectedVolume(initialVolume); // Restore previous volume when unmuting
    } else {
      setInitialVolume(selectedVolume); // Store the current volume before muting
      setSelectedVolume(0); // Set volume to 0 when muting
      synthRef.current.cancel(); // Stop ongoing TTS instantly when muting
    }
    setMuteTTS(!muteTTS);
  };


  // Click handler for selecting the man avatar
  const handleSelectManAvatar = () => {
    setSelectedAvatar('man');
    setVideoSource('ai-man.webm');
    // Change the video source when selecting the man avatar
    if (videoPlayerRef.current) {
      videoPlayerRef.current.src = process.env.PUBLIC_URL + '/ai-man.webm';
      videoPlayerRef.current.load(); // Load the new video source
    }
  };

  // Click handler for selecting the woman avatar
  const handleSelectWomanAvatar = () => {
    setSelectedAvatar('woman');
    setVideoSource('ai-woman.webm');
    // Change the video source when selecting the woman avatar
    if (videoPlayerRef.current) {
      videoPlayerRef.current.src = process.env.PUBLIC_URL + '/ai-woman.webm';
      videoPlayerRef.current.load(); // Load the new video source
    }
  };

  const handleAIResponse = () => {
    // Simulate AI response after a short delay
    setTimeout(() => {
      handleChatMessage('idle');
      const randomReply = getRandomLoremIpsum(); // API RESPONSE
      // Add AI message to the chat log
      const newAiMessageIndex = chatLog.length; // Calculate the index of the new AI message
      setChatLog((prevChatLog) => [...prevChatLog, { sender: 'AI', message: randomReply }]);
      // Update the aiLatestMessage state with the AI's response message
      setAiLatestMessage(randomReply);
      // Set the index of the latest AI message
      setLatestAiMessageIndex(newAiMessageIndex + 1);

      // Check if the AI's message contains 'video' to start the video player
      if (videoPlayerRef.current) {
        videoPlayerRef.current.play();
      }

      // Play text-to-speech for AI response
      handlePlayStopTextToSpeech(newAiMessageIndex + 1, randomReply); // Pass the index of the latest message

      handleChatMessage('play');
    }, 1000); // Simulate response delay (1 second in this example)
  }

  const handleSendMessage = () => {
    const combinedMessage = message.trim() || dictaphoneTranscript.trim(); // Use STT transcript if message is empty
   
    if (combinedMessage !== '' ) {
      // Add user message to the chat log
      setChatLog((prevChatLog) => [...prevChatLog, { sender: 'You', message: combinedMessage }]);
      setMessage('');
      setDictaphoneTranscript('');
      //handleAIResponse();
    }
    
    if (isListening) {
      setDictaphoneTranscript("");
      resetTranscript();
    }
  };  

  const getRandomLoremIpsum = () => {
    // Replace this with your own function to generate random responses
    const loremIpsums = [
      "How have you felt in the past week two weeks?",
      "In the past month, have you lost interest or pleasure in things you usually like to do, if yes, why?",
      "Have you felt sad, low, down, depressed, or hopeless recently? Why?",
      "Have you faced any difficulties lately?",
      "How would you describe your self-esteem?",
      "How is your sleep? Have you noticed any changes?",
      "How are your energy levels?",
      "Have others noticed you speaking or moving slower than usual lately?",
      "Have you noticed being restless or fidgety lately?",
      "Would you classify your appetite lately as normal?",
      "Do you blame yourself for a negative event that has recently transpired?",
      "Have you felt anger at anything recently? If so, what?",
      "Do you regularly have feelings of self-doubt or worthlessness?"
    ];
    return loremIpsums[Math.floor(Math.random() * loremIpsums.length)];
  };

  const handleTranscriptChange = (transcript) => {
    setDictaphoneTranscript(transcript);
  };

  const handlePlayStopTextToSpeech = (messageId, text) => {
    const isPlaying = textToSpeechStatus[messageId];

    if (isPlaying) {
      // Stop text-to-speech if it's already playing
      setVideoState('idle');
      synthRef.current.cancel();
      setTextToSpeechStatus((prevState) => ({
        ...prevState,
        [messageId]: false,
      }));
    } else {
      setVideoState('play');
      // Play text-to-speech for the given text
      setTextToSpeechStatus((prevState) => ({
        ...prevState,
        [messageId]: true,
      }));
      playTextToSpeech(text, selectedVoice, selectedPitch, selectedRate, selectedVolume, false)
        .then(() => {
          // Text-to-speech finished, reset the icon to "Play" state
          setTextToSpeechStatus((prevState) => ({
            ...prevState,
            [messageId]: false,
          }));
        })
        .catch(() => {
          // Text-to-speech interrupted, reset the icon to "Play" state
          setTextToSpeechStatus((prevState) => ({
            ...prevState,
            [messageId]: false,
          }));
        });    
    }
  
    // Handle video playback state
    if (videoState === 'play' && !isPlaying) {
      handlePause();
    } else if (videoState === 'idle' && isPlaying) {
      handlePlay();
    }
  };

  const toggleListening = (forceStop) => {
    if(forceStop) {
      SpeechRecognition.stopListening();
      setIsListening(false);
      return;
    }

    if (isListening) {
      SpeechRecognition.stopListening();
    } else {
      SpeechRecognition.startListening({ continuous: true });
    }
    setIsListening(!isListening);
  };

  if (!browserSupportsSpeechRecognition) {
    return <span>Browser doesn't support speech recognition.</span>;
  }
  
  const clearTranscript = (forceStop) => {
    toggleListening(forceStop);
    resetTranscript();
  };



  return (
    <Grid container spacing={2}>
      
      {/* Video Player Section (Left) */}
      <Grid item xs={12} md={6} style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-end' }}>
        <Paper sx={{ backgroundColor: 'unset', boxShadow: 'unset', marginBottom: '70px' }}>
          <div>
            <video ref={videoPlayerRef} style={{ maxWidth: '100%', height: 500}}>
              <source src={process.env.PUBLIC_URL + '/' + videoSource} />
            </video>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '10px' }}>
              <Tooltip title="Open Settings">
                <Button 
                  variant="contained" 
                  onClick={handleOpenSettings}
                  endIcon={<SettingsIcon />}
                  sx={{
                    backgroundColor: '#233036',
                    '&:hover': {
                      backgroundColor: '#447796',
                    },
                  }}
                >
                  Settings
                </Button>
              </Tooltip>
              <Tooltip title="Mute Video">
                <Button 
                  className="circularButton" 
                  onClick={handleToggleMuteTTS} 
                  startIcon={muteTTS ? <VolumeMuteIcon /> : <VolumeUpIcon />}
                />
              </Tooltip>
            </div>
          </div>
        </Paper>
      </Grid>

      {/* Chat Section (Right) */}
      <Grid item xs={12} md={6} style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <Paper sx={{backgroundColor: "unset", boxShadow: "unset", alignSelf: 'center'}}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '10px' }}>
            <WebcamCompStreamCapture handleAIResponse={handleAIResponse} handleSendMessage={handleSendMessage} toggleListening={toggleListening} clearTranscript={clearTranscript}/>
          </div>
        </Paper>
        <Paper sx={{backgroundColor: "unset", boxShadow: "unset"}}>
          <div className="ChatPage">
            <div className="chatContainer">
              <div className="chatLog" ref={chatLogRef}>
                {chatLog.length < 1 && <div className='startMessage'>Hey! I'm LUNA, ready to assist you on your mental health journey.</div>}
              
              {chatLog.map((chat, index) => (
                  <div className="row" style={{ display: 'flex' }} key={index}>
                    <div
                      className={`col-12 chatMessage ${chat.sender === 'AI' ? 'chatMessage-AI' : ''}`}
                      style={{ flex: '1', marginLeft: chat.sender === 'AI' ? 'auto' : '0' }}
                    >
                      {chat.sender === 'You' && <div className="senderLabel senderLabel-You">You</div>}
                      {chat.sender === 'AI' && <div className="senderLabel senderLabel-AI">AI</div>}
                      <span className={`chatBubble ${chat.sender === 'You' ? 'sender' : ''}`}>{chat.message}</span>
                      {chat.sender === 'AI' && (
                        <Tooltip title="Listen Again">
                          <Button
                            variant="text"
                            color="primary"
                            className="toggleTTSButton"
                            onClick={() => handlePlayStopTextToSpeech(index, chat.message)}
                          >
                          {textToSpeechStatus[index] ? <VolumeOffIcon /> : <VolumeUpIcon />}
                        </Button>   
                      </Tooltip>                 
                      )}
                      <hr />
                    </div>
                  </div>
                ))}
              </div>
              <div className="messageForm">
                <TextField
                  label="Type your message..."
                  fullWidth
                  value={message || dictaphoneTranscript}
                  onChange={(e) => setMessage(e.target.value)}
                  variant="outlined"
                  margin="normal"
                  className="messageInput"
                  onKeyPress={(event) => {
                    if (event.key === 'Enter') {
                      handleSendMessage();
                    }
                  }}
                  
                  InputProps={{
                    style: {color: 'white'},
                    endAdornment: (
                      <Tooltip title="Send Message">
                        <Button
                          variant="contained"
                          color="primary"
                          onClick={handleSendMessage}
                          endIcon={<SendIcon />}
                          sx={{
                            backgroundColor: '#233036',
                            '&:hover': {
                              backgroundColor: '#447796',
                            },
                          }}
                        >
                          Send
                        </Button>
                      </Tooltip>
                    ),
                  }}
                  InputLabelProps={{
                    style: {
                      color: 'white',
                    },
                  }}
                  sx={{
                    '& label.Mui-focused': {
                      color: '#447796', // Change label color on focus
                    },
                    '& .MuiOutlinedInput-root': {
                      borderRadius: '20px', // Make the border a bit more circular
                      borderColor: 'white', // Change outline color
                      borderWidth: '15px !important', // Increase border thickness
                      '&:hover fieldset': {
                        borderColor: 'white', // Change outline color on hover
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: 'white', // Change outline color on focus
                      },
                    },
                  }}
                />
              </div>
            </div>
          </div>
        </Paper>
      </Grid>
      
        {/* Settings Dialog */}
      <Dialog open={isSettingsOpen} onClose={() => handleCloseSettings(false)}>
          <DialogTitle>Text to Speech Settings</DialogTitle>
            <DialogContent>        
              <Box>
                {/* Add placeholders for images */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '20px' }}>
                  <div style={{ textAlign: 'center' }}>
                    <p>Man Avatar</p>
                    <img
                      src={process.env.PUBLIC_URL + '/man-ai.png'}
                      alt="Man Avatar"
                      style={{ width: '200px', height: '100px', cursor: 'pointer' }}
                      onClick={handleSelectManAvatar} // Attach the click handler
                    />
                  </div>
                  <div style={{ textAlign: 'center', marginTop: '10px' }}>
                    <p>Woman Avatar</p>
                    <img
                      src={process.env.PUBLIC_URL + '/woman-ai.png'}
                      alt="Woman Avatar"
                      style={{ width: '200px', height: '100px', cursor: 'pointer' }}
                      onClick={handleSelectWomanAvatar} // Attach the click handler
                    />
                  </div>
                </div>
                <FormControl fullWidth style={{marginTop: "20px"}}>
                  <InputLabel htmlFor="voice-select">Voice</InputLabel>
                  <Select
                    value={selectedVoiceTemp?.name}
                    onChange={handleVoiceChange}
                    label="Voice"
                    inputProps={{ id: 'voice-select' }}
                  >
                    {availableVoices.map((voice) => (
                      <MenuItem key={voice.name} value={voice.name}>
                        {voice.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <br />
                <Box sx={{ width: '100%', paddingTop: "20px" }}>
                  <InputLabel>Pitch</InputLabel>
                  <Slider
                    value={selectedPitchTemp}
                    min={0.5}
                    max={2}
                    step={0.1}
                    onChange={handlePitchChange}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(x) => x.toFixed(1)}
                  />
                </Box>
                <br />
                <Box sx={{ width: '100%' }}>
                  <InputLabel>Speed</InputLabel>
                  <Slider
                    value={selectedRateTemp}
                    min={0.5}
                    max={2}
                    step={0.1}
                    onChange={handleRateChange}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(x) => x.toFixed(1)}
                  />
                </Box>
                <br />
                <Box sx={{ width: '100%' }}>
                  <InputLabel>Volume</InputLabel>
                  <Slider
                    value={selectedVolumeTemp}
                    min={0}
                    max={1}
                    step={0.1}
                    onChange={handleVolumeChange}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(x) => x.toFixed(1)}
                  />
                </Box>
              </Box>
            </DialogContent>
          <DialogActions>
          <Button onClick={() => handleCloseSettings(true)} color="primary">
            Save
          </Button>
          <Button onClick={() => handleCloseSettings(false)} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};

export default ChatPage;