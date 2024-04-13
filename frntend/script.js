(function () {
	const query = "Hi Ella, my name is Julie Nguyen and I am interested in the Graduate program. I have a few questions about the program. Can you tell me more about the courses offered in the program? Also, I would like to know about the admission requirements and the application process. Thank you";

  const bodyElement = document.body;
  const btnLocation = ".a8X.gU"; //Toolbar button container
  const bodyLocation = ".gs"; //Body of email container
  const composeLocation = ".Am.aiL.aO9.Al.editable.LW-avf.tS-tW"; //Compose email container
  const imageURL = chrome.runtime.getURL("/logo.png");

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
      attributeFilter: ["href"],
    });
  }

  function injectButton() {
    const someDiv = document.querySelector(btnLocation);
    const btnCheck = document.querySelector("#capstone-button");
    if (someDiv && !btnCheck) {
      const container = document.createElement("div");
      container.classList.add("wG", "J-Z-I");

      const img = document.createElement("img");
      img.src = imageURL;
      img.alt = "Capstone Draft";
      img.id = "capstone-button";
      img.classList.add("aaA", "aMZ");

      img.style.cssText = "cursor: pointer; width: 20px; height: 20px";

      container.appendChild(img);
      document.querySelector(btnLocation).appendChild(container);
      img.addEventListener("click", handleClick());
    }
  }

  function handleClick() {
    document
      .getElementById("capstone-button")
      .addEventListener("click", function (event) {
        injectBody();
      });
  }

  //aer-llm api request
  function injectBody() {
    fetch(`http://127.0.0.1:8000?q=${query}`)
      .then((response) => response.json())
      .then((dta) => {
        if (document.querySelector(composeLocation)) {
          //document.querySelector(composeLocation).textContent = dta.response;
          typeOutResponse(dta.response);
        }
      })
      .catch((error) => alert(error));
  }
  // Typing out response character by character
  function typeOutResponse(response) {
    const composeContainer = document.querySelector(composeLocation);
    let index = 0;
    const interval = setInterval(function () {
      composeContainer.textContent += response[index++];
      if (index === response.length) {
        clearInterval(interval);
      }
    }, 10); // Adjust typing speed here
  }
  //Extracts text from email for future usage
  function extractBody() {
    const elements = document.querySelectorAll(bodyLocation);
    const extractedBody = [];
    elements.forEach((element) => {
      extractedBody.push(element.textContent.trim());
    });
    return extractedBody;
  }
  
  observePage();
  observeURLChanges();
})();
