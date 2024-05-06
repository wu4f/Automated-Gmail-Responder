(function () {
  let dotsInterval;
  const bodyElement = document.body;
  const btnLocation = '.a8X.gU'; //Toolbar button container
  const bodyLocation = '.gs'; //Body of email container
  const composeLocation = '.Am.aiL.aO9.Al.editable.LW-avf.tS-tW'; //Compose email container
  const imageURL = chrome.runtime.getURL('/logo.png');

  chrome.storage.local.get('savedText', function (data) {
    const defaultPrompt =
      'Task: Write an email response to the following email from a student with answers to their questions given the following context.';
    if (data.savedText === undefined) {
      chrome.storage.local.set({ savedText: defaultPrompt });
    }
  });

  //Checks for necessary elements to load
  const elementObserver = new MutationObserver(function (
    mutations,
    mutationInstance
  ) {
    const someDiv = document.querySelector(btnLocation);
    if (someDiv) {
      mutationInstance.disconnect();
      injectButton();
    }
  });

  function observePage() {
    elementObserver.observe(bodyElement, {
      childList: true,
      subtree: true,
    });
  }

  //Resets elementObserver when URL changes
  function observeURLChanges() {
    const locationObserver = new MutationObserver(function () {
      observePage();
    });

    locationObserver.observe(document, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['href'],
    });
  }

  function injectButton() {
    const someDiv = document.querySelector(btnLocation);
    const btnCheck = document.querySelector('#capstone-button');
    if (someDiv && !btnCheck) {
      const container = document.createElement('div');
      container.classList.add('wG', 'J-Z-I');

      const img = document.createElement('img');
      img.src = imageURL;
      img.alt = 'Capstone Draft';
      img.id = 'capstone-button';
      img.classList.add('aaA', 'aMZ');
      img.style.cssText = 'cursor: pointer; width: 20px; height: 20px';

      container.appendChild(img);
      someDiv.appendChild(container);
      img.addEventListener('click', handleClick());
    }
  }

  function handleClick() {
    document
      .getElementById('capstone-button')
      .addEventListener('click', function (event) {
        chrome.storage.local.get(['savedText'], function (result) {
          alert(result.savedText);
        });
        //injectBody();
      });
  }

  // Function to display loading dots in the text box
  function displayLoadingDots() {
    const composeContainer = document.querySelector(composeLocation);
    let dots = '';
    dotsInterval = setInterval(() => {
      dots += '.';
      composeContainer.textContent = dots;
      if (dots.length > 3) {
        dots = '';
      }
    }, 500);
  }

  // Function to stop displaying loading dots and set the actual response
  function stopLoadingDots() {
    clearInterval(dotsInterval);
  }

  //aer-llm api request
  function injectBody() {
    displayLoadingDots();
    let question = extractBody();
    fetch(`http://127.0.0.1:8000?q=${question}`)
      .then((response) => response.json())
      .then((dta) => {
        if (document.querySelector(composeLocation)) {
          stopLoadingDots();
          typeOutResponse(dta.response);
        }
      })
      .catch((error) => alert(error));
  }
  // Typing out response character by character
  function typeOutResponse(response) {
    const composeContainer = document.querySelector(composeLocation);
    if (!response) {
      console.log('No response from API');
      //injectBody(); //infinite loop?
      return;
    }
    let index = 0;
    let parser = '';
    const interval = setInterval(function () {
      parser += response[index++];
      composeContainer.innerHTML = parser;
      if (index === response.length) {
        clearInterval(interval);
      }
    }, 10); // Adjust typing speed here
  }
  //Extracts text from email for future usage
  function extractBody() {
    const elements = document.querySelectorAll(bodyLocation);
    let extractedBody = new Array();
    elements.forEach((element) => {
      Array.from(element.children).forEach((cont, index) => {
        extractedBody.push(cont.textContent.trim());
      });
    });
    return extractedBody.slice(2);
  }

  observePage();
  observeURLChanges();
})();
