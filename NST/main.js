import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
import * as faceapi from 'face-api.js';
tf.ENV.set('WEBGL_PACK', false);  // This needs to be done otherwise things run very slow v1.0.4
/**
 * Main application to start on window load
 */
class Main {
  constructor() {
    this.fileSelect = document.getElementById('file-select');

    // Initialize model selection
      this.loadMobileNetStyleModel().then(model => {
        this.styleNet = model;
      }).finally(() => this.enableStylizeButtons());

      this.loadOriginalTransformerModel().then(model => {
        this.transformNet = model;
      }).finally(() => this.enableStylizeButtons());
  

    this.initalizeWebcamVariables();
    this.initializeStyleTransfer();

    Promise.all([
      this.loadMobileNetStyleModel(),
      this.loadSeparableTransformerModel(),
    ]).then(([styleNet, transformNet]) => {
      console.log('Loaded styleNet');
      this.styleNet = styleNet;
      this.transformNet = transformNet;
      this.enableStylizeButtons()
    });
  }

  async loadMobileNetStyleModel() {
    if (!this.mobileStyleNet) {
      this.mobileStyleNet = await tf.loadGraphModel(
        'saved_model_style_js/model.json');
    }

    return this.mobileStyleNet;
  }

  async loadInceptionStyleModel() {
    if (!this.inceptionStyleNet) {
      this.inceptionStyleNet = await tf.loadGraphModel(
        'saved_model_style_inception_js/model.json');
    }
    
    return this.inceptionStyleNet;
  }

  async loadOriginalTransformerModel() {
    if (!this.originalTransformNet) {
      this.originalTransformNet = await tf.loadGraphModel(
        'saved_model_transformer_js/model.json'
      );
    }

    return this.originalTransformNet;
  }

  async loadSeparableTransformerModel() {
    if (!this.separableTransformNet) {
      this.separableTransformNet = await tf.loadGraphModel(
        'saved_model_transformer_separable_js/model.json'
      );
    }

    return this.separableTransformNet;
  }

  initalizeWebcamVariables() {
    this.contentImg = document.getElementById('content-img');
    this.snapButton = document.getElementById('snap-button');
    this.snapButton.onclick = () => {
      const hiddenCanvas = document.getElementById('hidden-canvas');
      const hiddenContext = hiddenCanvas.getContext('2d');
      hiddenCanvas.width = 320;
      hiddenCanvas.height = 320;
      hiddenContext.drawImage(this.webcamVideoElement, 0, 0, 
        hiddenCanvas.width, hiddenCanvas.height);
      const imageDataURL = hiddenCanvas.toDataURL('image/jpg');
      this.contentImg.src = imageDataURL;
    };
    this.webcamVideoElement = document.getElementById('webcam-video');

    navigator.getUserMedia = navigator.getUserMedia ||
        navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;
      navigator.getUserMedia(
        {
          video: true
        },
        (stream) => {
          this.stream = stream;
          this.webcamVideoElement.srcObject = stream;
          this.webcamVideoElement.play();
        },
        (err) => {
          console.error(err);
        }
      );
  }



  initializeStyleTransfer() {
    // Initialize images
    this.contentImg = document.getElementById('content-img');
    this.contentImg.onerror = () => {
      alert("Error loading " + this.contentImg.src + ".");
    }
    this.styleImg = document.getElementById('style-img');
    this.styleImg.onerror = () => {
      alert("Error loading " + this.styleImg.src + ".");
    }
    this.stylized = document.getElementById('stylized');

    // Initialize images
    
    this.styleRatio = 1.0

    // Initialize buttons
    this.styleButton = document.getElementById('style-button');
    this.styleButton.onclick = () => {
      this.disableStylizeButtons();
      this.startStyling().finally(() => {
        this.enableStylizeButtons();
      });
    };
    // Initialize selector
    //this.styleSelect = document.getElementById('style-select');
    //this.styleSelect.onchange = (evt) => this.setImage(this.styleImg, evt.target.value);
   // this.styleSelect.onclick = () => this.styleSelect.value = '';
    //document.addEventListener('click', function (event) {
      //if (!event.target.matches('.style-img')) return;
      // Don't follow the link
      //event.preventDefault();
      //this.styleImg = document.getElementById('style-img');
      
      //this.styleImg.src = 'images/' + event.target.getAttribute('img-name') + '.jpg';
    //}, false);
    $(".style-img").on("click", function(){
      this.styleImg = document.getElementById('style-img');
      this.styleImg.src = 'images/' + event.target.getAttribute('img-name') + '.jpg';
      $(".style-img").removeClass("selected");
      $(this).addClass("selected");
    })
   $(document).on('keypress',function(event) {
      if(event.which==53)$('#snap-button').click();
      if(event.which==54)$('#style-button').click();
      if(event.which==55)changeModel('nst');
    })
  }

  // Helper function for setting an image
  setImage(element, selectedValue) {
    element.src = 'images/' + selectedValue + '.jpg';
  }
  
  enableStylizeButtons() {
    this.styleButton.disabled = false;
    this.styleButton.textContent = 'Stylize';
  }

  disableStylizeButtons() {
    this.styleButton.disabled = true;
  }

  async startStyling() {
    await tf.nextFrame();
    this.styleButton.textContent = 'Generating style representation';
    await tf.nextFrame();
    let bottleneck = await tf.tidy(() => {
      return this.styleNet.predict(tf.browser.fromPixels(this.styleImg).toFloat().div(tf.scalar(255)).expandDims());
    })
    if (this.styleRatio !== 1.0) {
      this.styleButton.textContent = 'Generating identity style representation';
      await tf.nextFrame();
      const identityBottleneck = await tf.tidy(() => {
        return this.styleNet.predict(tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims());
      })
      const styleBottleneck = bottleneck;
      bottleneck = await tf.tidy(() => {
        const styleBottleneckScaled = styleBottleneck.mul(tf.scalar(this.styleRatio));
        const identityBottleneckScaled = identityBottleneck.mul(tf.scalar(1.0-this.styleRatio));
        return styleBottleneckScaled.addStrict(identityBottleneckScaled)
      })
      styleBottleneck.dispose();
      identityBottleneck.dispose();
    }
    this.styleButton.textContent = 'Stylizing image...';
    await tf.nextFrame();
    const stylized = await tf.tidy(() => {
      return this.transformNet.predict([tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims(), bottleneck]).squeeze();
    })
    await tf.browser.toPixels(stylized, this.stylized);
    bottleneck.dispose();  // Might wanna keep this around
    stylized.dispose();
  }

  async startCombining() {
    await tf.nextFrame();
    await tf.nextFrame();
    const bottleneck1 = await tf.tidy(() => {
      return this.styleNet.predict(tf.browser.fromPixels(this.combStyleImg1).toFloat().div(tf.scalar(255)).expandDims());
    })
    await tf.nextFrame();
    const bottleneck2 = await tf.tidy(() => {
      return this.styleNet.predict(tf.browser.fromPixels(this.combStyleImg2).toFloat().div(tf.scalar(255)).expandDims());
    });
    await tf.nextFrame();
    const combinedBottleneck = await tf.tidy(() => {
      const scaledBottleneck1 = bottleneck1.mul(tf.scalar(1-this.combStyleRatio));
      const scaledBottleneck2 = bottleneck2.mul(tf.scalar(this.combStyleRatio));
      return scaledBottleneck1.addStrict(scaledBottleneck2);
    });

    const stylized = await tf.tidy(() => {
      return this.transformNet.predict([tf.browser.fromPixels(this.combContentImg).toFloat().div(tf.scalar(255)).expandDims(), combinedBottleneck]).squeeze();
    })
    await tf.browser.toPixels(stylized, this.combStylized);
    bottleneck1.dispose();  // Might wanna keep this around
    bottleneck2.dispose();
    combinedBottleneck.dispose();
    stylized.dispose();
  }

  async benchmark() {
    const x = tf.randomNormal([1, 256, 256, 3]);
    const bottleneck = tf.randomNormal([1, 1, 1, 100]);

    let styleNet = await this.loadInceptionStyleModel();
    let time = await this.benchmarkStyle(x, styleNet);
    styleNet.dispose();

    styleNet = await this.loadMobileNetStyleModel();
    time = await this.benchmarkStyle(x, styleNet);
    styleNet.dispose();

    let transformNet = await this.loadOriginalTransformerModel();
    time = await this.benchmarkTransform(
        x, bottleneck, transformNet);
    transformNet.dispose();

    transformNet = await this.loadSeparableTransformerModel();
    time = await this.benchmarkTransform(
      x, bottleneck, transformNet);
    transformNet.dispose();

    x.dispose();
    bottleneck.dispose();
  }

  async benchmarkStyle(x, styleNet) {
    const profile = await tf.profile(() => {
      tf.tidy(() => {
        const dummyOut = styleNet.predict(x);
        dummyOut.print();
      });
    });
    console.log(profile);
    const time = await tf.time(() => {
      tf.tidy(() => {
        for (let i = 0; i < 10; i++) {
          const y = styleNet.predict(x);
          y.print();
        }
      })
    });
    console.log(time);
  }

  async benchmarkTransform(x, bottleneck, transformNet) {
    const profile = await tf.profile(() => {
      tf.tidy(() => {
        const dummyOut = transformNet.predict([x, bottleneck]);
        dummyOut.print();
      });
    });
    console.log(profile);
    const time = await tf.time(() => {
      tf.tidy(() => {
        for (let i = 0; i < 10; i++) {
          const y = transformNet.predict([x, bottleneck]);
          y.print();
        }
      })
    });
    console.log(time);
  }
}

window.addEventListener('load', () => new Main());
