// TalkingAvatar.js
import React, { useState, useEffect } from 'react';
import { Avatar } from 'talking_avatar';

const TalkingAvatar = ({ message, isAI }) => {
  const [isTalking, setIsTalking] = useState(false);

  // Trigger animation when a new message is received
  useEffect(() => {
    setIsTalking(true);
    const timeout = setTimeout(() => {
      setIsTalking(false);
    }, 1000); // Set the duration of the talking animation here (in milliseconds)
    return () => clearTimeout(timeout);
  }, [message]);

  return (
    <Avatar
      width={100}
      height={100}
      eyeType={isTalking ? 'talking' : 'default'}
      eyebrowType={isAI ? 'raised' : 'default'}
    />
  );
};

export default TalkingAvatar;
