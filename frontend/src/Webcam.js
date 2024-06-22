import React, { useState } from "react";
import "./ChatPage.css"; // Import your custom CSS file for styling
import WebcamStreamCapture from './WebcamStreamCapture';



const WebcamComp = () => {
  return (
    <div>
      <WebcamStreamCapture/>
    </div>
  );
};

export default WebcamComp;
