import { DragEvent, useState, useEffect } from "react";

const text_style = {
    color: 'black',
    fontSize: '1rem',
};

const hover_effect = {
    backgroundColor: 'rgba(0,0,0,0.3)',
    color: 'white',
    fontSize: '24px',
    border: '0.2rem dashed',
};

const text_preview_style = {
  paddingTop: '1rem',
  width: '60%',
  margin: '0 auto',
};

interface ImageUploadProps {
     onFile: (file: File) => void;
}

export default function FileDrop({ onFile }: ImageUploadProps) {
  const [isOver, setIsOver] = useState(false);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [displayedText, setDisplayedText] = useState('');

  const animateText = (text: string) => {
    let displayed = '';
    const speed = (i: number) => 100 + i * 25; // Change 25 to lower for higher speed

    for (let i = 0; i < text.length; i++) {
        setTimeout(() => {
            displayed += text[i];
            setDisplayedText(displayed);
        }, speed(i));
    }
};

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsOver(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsOver(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsOver(false);

    // Process the first .txt file dropped
    const file = Array.from(event.dataTransfer.files).find(file => file.type === "text/plain");

    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFileContent(e.target?.result as string);
        animateText(e.target?.result as string);
        onFile(file);
      };
      reader.readAsText(file);
    }
  };

  useEffect(() => {
    // Attach the drag and drop handlers to the entire document
    document.addEventListener('dragover', handleDragOver as any);
    document.addEventListener('dragleave', handleDragLeave as any);
    document.addEventListener('drop', handleDrop as any);

    return () => {
      document.removeEventListener('dragover', handleDragOver as any);
      document.removeEventListener('dragleave', handleDragLeave as any);
      document.removeEventListener('drop', handleDrop as any);
    };
  }, []);

  return (
    <div>
      <div
          style={{
              display: isOver ? "flex" : "none",
              justifyContent: "center",
              alignItems: "center",
              height: "95vh",
              width: "95vw",
              position: "fixed",
              top: "2.5vh",
              left: "2.5vw",
              zIndex: 9999,
              ...hover_effect,
          }}
      >
          {isOver && <div style={text_style}>Drop a .txt file here to upload</div>}
      </div>
      <div style={text_preview_style}>
        {/* Display the content of the text file */}
        <p>{displayedText}</p>
      </div>
    </div>
  );
}
