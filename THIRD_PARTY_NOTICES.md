# ECG-Agent

The ECG measurement implementation is derived from the numerical routines in
`gustmd0121/ECG-Agent`, revision `0ab07899ec0a39b087bebd560b280cad484eb0d0`,
`src/medrax/tools/classification.py`. Input validation, structured output, status
handling and beat-aligned QTc pairing are adapted in this repository.

Source: https://github.com/gustmd0121/ECG-Agent/tree/0ab07899ec0a39b087bebd560b280cad484eb0d0

MIT License

Copyright (c) 2026 Hyunseung Chung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# PTB-XL demonstration and benchmark records

The optional demonstration and benchmark downloaders use public PTB-XL 1.0.3 records:
Wagner, P., Strodthoff, N., Bousseljot, R.-D., Samek, W., and Schaeffter, T.
*PTB-XL, a large publicly available electrocardiography dataset* (version 1.0.3).
PhysioNet. https://doi.org/10.13026/kfzx-aw45

Dataset: https://physionet.org/content/ptb-xl/1.0.3/
License: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/

The downloader converts the published 16-bit WFDB samples into physical-unit MAT
data using header gains and baselines. Original bytes, checksums, and attribution
are retained in the ignored local demonstration bundle. Raw ECG data is not
included in the source distribution.

# ECGFounder

The optional architecture and ordered task labels are extracted from
PKUDigitalHealth/ECGFounder at revision
`04edac702b61c91face519774ddcc0cd712fef23`. Training code, dataset classes, and
unused imports are omitted. The inference architecture and parameter names are
preserved. The adapter follows the pinned PTB-XL 150-class evaluator's global
normalization without filtering, distinct from its downstream LVEF preprocessing.

Source: https://github.com/PKUDigitalHealth/ECGFounder/tree/04edac702b61c91face519774ddcc0cd712fef23
Weights: https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/d9b1793951b2342f5f7e84f1ac03cd37f8a08724
Source and model repositories declare MIT licensing. Weights are downloaded
separately and are not included in the source distribution.

MIT License
Copyright (c) 2025 PKUDigitalHealth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
