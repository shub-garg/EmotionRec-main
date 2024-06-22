import React, { useRef, useState, useEffect } from 'react';

const VideoPlayer = ({ videoState }) => {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false); // Set initial state to false

  // Function to handle video play/pause based on the videoState prop
  useEffect(() => {
    const video = videoRef.current;
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
  }, [videoState]);

  const handlePlay = () => {
    const video = videoRef.current;
    video.play().then(() => {
      setIsPlaying(true);
    }).catch((error) => {
      console.error('Error playing video:', error);
    });
  };

  const handlePause = () => {
    const video = videoRef.current;
    video.pause();
    setIsPlaying(false);
  };

  const handleReset = () => {
    const video = videoRef.current;
    video.currentTime = 0;
    video.play().then(() => {
      setIsPlaying(true);
    }).catch((error) => {
      console.error('Error playing video:', error);
    });
  };

  return (
    <div>
      <video ref={videoRef} style={{ maxWidth: 500, maxHeight: 500, padding: 20 }}>
        <source src={process.env.PUBLIC_URL + '/video-name.mp4'} type="video/mp4" />
      </video>
      <div>
        {!isPlaying ? (
          <button onClick={handlePlay}>Play</button>
        ) : (
          <button onClick={handlePause}>Pause</button>
        )}
        <button onClick={handleReset}>Reset</button>
      </div>
    </div>
  );
};

export default VideoPlayer;
