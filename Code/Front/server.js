const express = require('express');
const fs = require('fs');
const cors = require('cors');
const https = require('https');  // Import https module
const app = express();

// Use cors middleware and allow all origins
app.use(cors({ origin: '*' }));
app.options('*', cors());

app.use(express.json());

app.post('/save-text', (req, res) => {
    const userName = req.body.text1;
    const userEmail = req.body.text2;
    const fileContent1 = req.body.text3;
    const fileContent2 = req.body.text4;

    console.log(`Received Name: ${userName}`);
    console.log(`Received Email: ${userEmail}`);

    // Concatenate
    const combinedfilename = userName + '\n' + userEmail;
    const combinedContent = userName + '\n' + userEmail +'\n' + fileContent1 + '\n' + fileContent2;

    // Write the combinedContent to a file with fileName1
    fs.writeFile(`${combinedfilename}.txt`, combinedContent, err => {
        if (err) {
            console.error(err);
            res.status(500).json({ error: 'Failed to save file' });
        } else {
            res.status(200).json({ message: 'File saved successfully' });
        }
    });
});

const port = 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});