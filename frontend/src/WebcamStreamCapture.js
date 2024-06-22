import React, { useState } from "react";
import Webcam from "react-webcam";
import IconButton from "@mui/material/IconButton";
import VideoCameraFrontIcon from '@mui/icons-material/VideoCameraFront';
import CancelIcon from "@mui/icons-material/Cancel";
import {  Button, Tooltip } from '@mui/material';
import './ChatPage.css';

const fn = async (video) => {
  const formData = new FormData();
    console.log('File size:', video.size);
    formData.append('video', video, 'react-webcam-stream-capture.webm');
    // formData.append('text', video, 'react-webcam-stream-capture.webm');

    try {
      const response = await fetch('http://localhost:8080/app', {
        method: 'POST',
        body: formData,
      });


      if (response.ok) {

        // console.log("Hello", response)
        const responseData = await response.json();
        const { labels, maximum, text_prediction } = responseData;
  
        console.log('Labels:', labels);
        console.log('Maximum:', maximum);
        // const max_value = { labels.data}
        alert('Video uploaded successfully');
        alert('Maximum:' + JSON.stringify(maximum));
        alert('Labels:' + JSON.stringify(labels));
        alert('Text_Prediction:' + JSON.stringify(text_prediction));
      } else {
        alert('Upload failed');
      }
    } catch (error) {
      alert('Error while uploading');
      console.error('Upload error:', error);
    }
}

const WebcamStreamCapture = ({ toggleListening, handleAIResponse, handleSendMessage, clearTranscript }) => {
    const webcamRef = React.useRef(null);
    const mediaRecorderRef = React.useRef(null);
    const [capturing, setCapturing] = React.useState(false);
    const [recordedChunks, setRecordedChunks] = React.useState([]);
    const [isMinimized, setIsMinimized] = useState(false);

  
    // Function to toggle the webcam on and off
    const toggleWebcam = () => {
      setIsMinimized(!isMinimized);
    };
  
    // Function to close the webcam
    const closeWebcam = () => {
      setIsMinimized(true);
    };
  
    const handleVoiceAndVideoCapture = () =>{
        handleStartCaptureClick(true);
    }
      
    const videoConstraints = {
      width: 200,
      height: 200,
      facingMode: "user"
    };

    const handleStartCaptureClick = React.useCallback((wantVoice) => {
      if(wantVoice){
        toggleListening();
      }
      setCapturing(true);
      handleAIResponse();

      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: "video/webm"
      });
      mediaRecorderRef.current.addEventListener(
        "dataavailable",
        handleDataAvailable
      );
      mediaRecorderRef.current.start();

    }, [webcamRef, setCapturing, mediaRecorderRef]);
  
    const handleDataAvailable = React.useCallback(
      ({ data }) => {
        if (data.size > 0) {
          setRecordedChunks((prev) => prev.concat(data));
        }
      },
      [setRecordedChunks]
    );
  
    const handleStopCaptureClick = React.useCallback(() => {
      mediaRecorderRef.current.stop();
      clearTranscript(true);
      setCapturing(false);
      // const blob = new Blob(recordedChunks, {
      //   type: "video/webm",
      //   // filename: "react-webcam-stream-capture.webm"
      // });
      console.log(recordedChunks)
      if (recordedChunks.length) {
        const blob = new Blob(recordedChunks, {
          type: "video/webm"
        });
        console.log(blob);
        console.log(blob.size);
        fn(blob);
      }

    }, [mediaRecorderRef, webcamRef, setCapturing, capturing, recordedChunks]);

    const handleStop = () => {
      handleSendMessage();
    };
  
    const handleDownload = React.useCallback(() => {
      if (recordedChunks.length) {
        const blob = new Blob(recordedChunks, {
          type: "video/webm"
        });
        fn(blob);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        document.body.appendChild(a);
        a.style = "display: none";
        a.href = url;
        a.download = "react-webcam-stream-capture.webm";
        a.click();
        window.URL.revokeObjectURL(url);
        setRecordedChunks([]);
        console.log(blob, url, a, recordedChunks)
      }
      
    }, [recordedChunks]);
  
    return (
      <div>
        <div className={isMinimized? "webcam-container webcome-circle-close" : "webcam-container webcome-circle-open" }>
          {isMinimized ? (
            <IconButton aria-label="Maximize Webcam" onClick={toggleWebcam}>
              <VideoCameraFrontIcon/>
            </IconButton>
          ) : (
            <>
            <div style={{height: "200px"}}>
              <Webcam audio={true} ref={webcamRef} muted = {true} mirrored={true}/>
            </div>
            <IconButton
                aria-label="Close Webcam"
                onClick={closeWebcam}
                style={{
                  position: 'absolute',
                  top: 80,
                  right: -10,
                  backgroundColor: 'white',
                  borderRadius: '50%'
                }}
              >
                <CancelIcon />
              </IconButton>
            </>
          )}
        </div>
        <div style={{ display: "flex", justifyContent: "space-around", gap: "20px"}}>
          {capturing ? (
              <Tooltip title="Stop Capture">
                <Button
                  sx={{
                    backgroundColor: '#233036',
                    '&:hover': {
                      backgroundColor: '#447796',
                    },
                  }}
                  variant="contained"
                  onClick={() => 
                    {
                      handleStopCaptureClick();
                      handleStop();
                    }
                  }
                >
                  Stop Capture
              </Button>
              </Tooltip>
            ) : (
              <>
              <Tooltip title="Start Capture">
                <Button
                  sx={{
                    backgroundColor: '#233036',
                    '&:hover': {
                      backgroundColor: '#447796',
                    },
                  }}
                  variant="contained"
                  onClick={() => handleStartCaptureClick(false)}
                  >
                  Record Video
              </Button>
              </Tooltip>
              <Tooltip title="Start Capture (With Audio)">
                <Button
                  sx={{
                    backgroundColor: '#233036',
                    '&:hover': {
                      backgroundColor: '#447796',
                    },
                  }}
                  variant="contained"
                  onClick={handleVoiceAndVideoCapture}
                >
                  Record Video + Audio
              </Button>
              </Tooltip>
              </>
            )}
            {recordedChunks.length > 0 && (
              <Tooltip title="Download">
                <Button
                  sx={{
                    backgroundColor: '#233036',
                    '&:hover': {
                      backgroundColor: '#447796',
                    },
                  }}
                  variant="contained"
                  onClick={handleDownload}
                >
                  Download
                </Button>
            </Tooltip>
          )}
        </div>
      </div>
    );
  };

  export default WebcamStreamCapture;