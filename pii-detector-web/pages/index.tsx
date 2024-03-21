import { Flex, Text, Button, TextField, TextArea, Badge } from '@radix-ui/themes';
import FileDrop from './components/dragUpload';

const title_style = {
  color: 'black', // Makes text black
  fontSize: '24px', // Larger text
  fontWeight: 'bold', // Bold text
};

const main_content = {
  paddingTop: '5rem',
  paddingRight: '5rem',
  paddingLeft: '5rem',
};

const label_style = {
  color: 'grey', // Makes text grey
  fontSize: '14px', // Smaller text
};

export default function mainPage() {
  return (
    <div>
      <div>
        <Flex direction="column" align="center" style={main_content}>
          <Text style={title_style}>PII Detector</Text>
          <Text style={label_style}>Upload a file to detect PII</Text>
          <FileDrop onFiles={(files) => console.log(files)} onRemove={(index) => console.log(index)} />
        </Flex>
      </div>
      {/* <div>
        <FileDrop onFiles={(files) => console.log(files)} onRemove={(index) => console.log(index)} />
      </div> */}
    </div>
  )
}