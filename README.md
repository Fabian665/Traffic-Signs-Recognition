# Traffic-Signs-Recognition

<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Fabian665/Traffic-Signs-Recognition">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Traffic Signs Recognition</h3>

  <p align="center">
    Python traffic signs recognition built for BGR admission
    <br />
    <br />
    <a href="https://github.com/Fabian665/Traffic-Signs-Recognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/Fabian665/Traffic-Signs-Recognition/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is an admission to the BGRacing team by Roey Fabian 

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Python](https://python.org/)
* [TensorFlow](https://tensorflow.org/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

* opencv-python~=4.5.5.62
* numpy~=1.19.3
* tensorflow~=2.4.0
* imageai~=2.1.6
* kaggle~=1.5.12
* pandas~=1.0.5
* scikit-learn~=1.0.2
* matplotlib~=3.3.2  


### Installation

1. Get a kaggle API Key at [kaggle.com](https://www.kaggle.com/) and save it under C:\users\\{your_user_name}\\.kaggle 
2. Clone the repo
   ```sh
   git clone https://github.com/Fabian665/Traffic-Signs-Recognition.git
   ```
3. Run setup.py
   ```sh
   python setup.py
   ```
4. Run train.py or upload the model to models directory
   ```sh
   python train.py
   ```
5. If you want to see the accuracy of the model run analyze_model.py
   ```sh
   python analyze_model.py
   ```



<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


###rsd.py


Start detecting traffic signs using rsd.py
```sh
python rsd.py {file_path}
```
You can also use in your own code
```python
import os
from rsd import RoadSignsDetection
from utils import get_image

image = get_image(os.path.join('images', 'v1', 'example1.png'))
app = RoadSignsDetection()
app.predict_one(image)
```
Or for multiple images
```python
import os
from rsd import RoadSignsDetection
from utils import get_images_from_dir

image_list = get_images_from_dir(os.path.join('images', 'v1'))
app = RoadSignsDetection()
app.predict_list(image_list)
``` 


###rsdv2.py


rsdv2.py detects stop signs from images that has not been cropped to the sign itself

to use it you can skip all the steps above
1. Create models directory and place "resnet50_coco_best_v2.1.0.h5" in it
2. Start detecting stop signs using rsdv2.py
   ```sh
   python rsd.py images\v1\example1.png
   ```
   You can also use in your own code
   ```python
   import os
   from rsdv2 import StopSignsDetection
   
   app = StopSignsDetection()
   app.detect(os.path.join('images', 'v2', 'example.jpg'))
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Roey Fabian - fabian665@gmail.com

Project Link: [https://github.com/Fabian665/Traffic-Signs-Recognition](https://github.com/Fabian665/Traffic-Signs-Recognition)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [imageai](https://github.com/OlafenwaMoses/ImageAI)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Fabian665/Traffic-Signs-Recognition.svg?style=for-the-badge
[contributors-url]: https://github.com/Fabian665/Traffic-Signs-Recognition/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Fabian665/Traffic-Signs-Recognition.svg?style=for-the-badge
[forks-url]: https://github.com/Fabian665/Traffic-Signs-Recognition/network/members
[stars-shield]: https://img.shields.io/github/stars/Fabian665/Traffic-Signs-Recognition.svg?style=for-the-badge
[stars-url]: https://github.com/Fabian665/Traffic-Signs-Recognition/stargazers
[issues-shield]: https://img.shields.io/github/issues/Fabian665/Traffic-Signs-Recognition.svg?style=for-the-badge
[issues-url]: https://github.com/Fabian665/Traffic-Signs-Recognition/issues
[license-shield]: https://img.shields.io/github/license/Fabian665/Traffic-Signs-Recognition.svg?style=for-the-badge
[license-url]: https://github.com/Fabian665/Traffic-Signs-Recognition/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/fabian-roey
[product-screenshot]: images/screenshot.png