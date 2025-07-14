// Import Transformers.js - Updated import
if (!navigator.gpu) {
  document.body.innerHTML = `
          <div style="color: red; text-align: center; margin-top: 3em; font-size: 1.3em;">
            <b>WebGPU is not supported in this browser.</b><br>
            Please use a recent version of Chrome or Edge with WebGPU enabled.<br>
            <a href="https://webgpureport.org/" target="_blank" style="color: #FFD21F;">Check your browser's WebGPU support here</a>.
          </div>
        `;
  throw new Error("WebGPU not supported");
}

// for llama
import {
  AutoTokenizer,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.min.js";

// // // for gpt 2 (FALLBACK)
// import {
// //   AutoTokenizer,
// //   pipeline,
// // } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js";

// AI Model Management
let generator = null;
let isModelLoaded = false;
let currentModelName = "";

// Initialize the AI model
async function initializeAI() {
  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingStatus = document.getElementById("loadingStatus");

  try {
    loadingOverlay.style.display = "flex";
    loadingStatus.textContent = "Loading GPT-2 Medium model...";

    // Load the text generation pipeline with GPT-2 Medium
    generator = await pipeline(
      "text-generation",
      // "openai-community/gpt2-medium",
      "onnx-community/Llama-3.2-1B-Instruct-q4f16",
      {
        quantized: true,
        device: "webgpu",
        progress_callback: (data) => {
          if (data.status === "progress") {
            loadingStatus.textContent = `Loading: ${Math.round(
              data.progress || 0
            )}%`;
          }
        },
      }
    );
    currentModelName = "LLAMA 3.2 1B";
    isModelLoaded = true;
    loadingStatus.textContent = "Model loaded successfully!";
    console.log("AI model loaded successfully");
    console.log("LLAMA 3.2 1B loaded via Transformers.js.");
    setTimeout(() => {
      loadingOverlay.style.display = "none";
    }, 1000);
  } catch (error) {
    console.error("Failed to load AI model:", error);
    loadingStatus.textContent =
      "Failed to load model. Using fallback responses.";
    // Try alternative lightweight model
    try {
      loadingStatus.textContent = "Trying alternative model...";
      generator = await pipeline("text-generation", "Xenova/gpt2", {
        quantized: true,
        device: "webgpu",
        progress_callback: (data) => {
          if (data.status === "progress") {
            loadingStatus.textContent = `Loading GPT-2: ${Math.round(
              data.progress || 0
            )}%`;
          }
        },
      });
      currentModelName = "GPT-2";
      isModelLoaded = true;
      loadingStatus.textContent = "Alternative model loaded!";
      console.log("Alternative AI model loaded successfully");
      setTimeout(() => {
        loadingOverlay.style.display = "none";
      }, 1000);
    } catch (fallbackError) {
      console.error("Fallback model also failed:", fallbackError);
      setTimeout(() => {
        loadingOverlay.style.display = "none";
      }, 2000);
    }
  }
}


async function generateAIResponse(
  fighterType,
  topic,
  conversationHistory = []
) {
  if (!isModelLoaded || !generator) {
    console.log("Model not loaded, using fallback");
    return getFallbackResponse(fighterType, topic);
  }

  try {
    // Get stance
    const stance = getStanceForFighter(fighterType, topic);

    // Only include the last message (if any)
    const lastMessage =
      conversationHistory.length > 0
        ? conversationHistory[conversationHistory.length - 1]
        : null;

    // FIXED: Proper prompt for Llama 3.2 1B - MOM VS MOM FIGHT
    const prompt = `You are an Asian mother in a heated argument with ANOTHER Asian mother about "${topic}".
Your stance: You ${stance} "${topic}" and will never back down.

FIGHT STYLE:
- You're arguing with an EQUAL, not lecturing a child
- Attack their logic, not their housekeeping
- Use "aiyah," "walao," "haiya" for dramatic effect
- Be sharp-tongued and fierce
- Focus ONLY on "${topic}" - no other topics
- Keep responses short and punchy (1-2 sentences max)

FIGHTING TACTICS:
- Question their intelligence about "${topic}"
- Bring up whose family/friends are smarter about this
- Challenge their experience or knowledge
- Use dramatic Asian mom expressions
- Be competitive and never admit defeat

Examples of mom-vs-mom fighting:
- "Aiyah! You don't know anything about this! My sister-in-law already proved you wrong!"
- "Walao! How can you be so stubborn? Even my mother-in-law smarter than you!"
- "Haiya! You talking nonsense! My friend's doctor said completely different!"

Last message from opponent: ${lastMessage ? `"${lastMessage.text}"` : "None"}

Your fierce response about "${topic}":`;

    console.log("Generating Asian mom response with prompt:", prompt);

    const result = await generator(prompt, {
      max_new_tokens: 80, // Shorter to avoid repetition
      temperature: 0.9, // Higher for more variety
      top_p: 0.85,
      do_sample: true,
      repetition_penalty: 1.4, // Higher to prevent repetition
      pad_token_id: 50256,
      no_repeat_ngram_size: 3, // Prevent 3-word repetitions
    });

    console.log("Raw AI result:", result);

    // Clean up response
    let response = result[0].generated_text;
    response = response.replace(prompt, "").trim();

    // Take first few sentences, stop at natural break
    const sentences = response.split(/[.!?]+/);
    response = sentences.slice(0, 3).join(". ");

    // Ensure proper punctuation
    if (!response.match(/[.!?]$/)) {
      response += "!";
    }

    // Remove quotes and extra whitespace
    response = response.replace(/^["\s]+|["\s]+$/g, "");

    console.log("Final Asian mom response:", response);
    return response;
  } catch (error) {
    console.error("AI generation failed:", error);
    return getAsianMomFallback(
      fighterType,
      topic,
      getStanceForFighter(fighterType, topic)
    );
  }
}

// Enhanced fallback with MOM VS MOM fighting
function getAsianMomFallback(fighterType, topic, stance) {
  const templates = {
    "COMPLETELY FOR": [
      `Aiyah! You don't understand ${topic} at all! My neighbor already told me you're wrong!`,
      `Walao! How can you be so blur about ${topic}? Even my mother-in-law knows better!`,
      `Haiya! You talking nonsense about ${topic}! My friend's sister proved it works!`,
      `${topic} is obviously good! Aiyoh! Your brain not working or what?`,
      `Listen here! ${topic} is amazing and you just jealous! My cousin's wife already successful with it!`,
    ],
    "VIOLENTLY AGAINST": [
      `Aiyah! ${topic} is complete rubbish! My sister-in-law warned me about people like you!`,
      `Walao! How can you support ${topic}? Even my hairdresser knows it's no good!`,
      `Haiya! ${topic} is disaster waiting to happen! You never learn from other people's mistakes!`,
      `${topic} is terrible! Aiyoh! My friend's doctor said stay away from this kind of thing!`,
      `No way! ${topic} is bad news! Eh! You want to argue with my experience?`,
    ],
  };

  const stanceTemplates = templates[stance] || templates["COMPLETELY FOR"];
  return stanceTemplates[Math.floor(Math.random() * stanceTemplates.length)];
}

// Helper function to determine stance (pro/against) for fighter
function getStanceForFighter(fighterType, topic) {
  // You can customize this logic based on your app's needs
  // For now, alternating or random assignment
  const stances = ["COMPLETELY FOR", "VIOLENTLY AGAINST"];

  // Simple hash-based assignment to ensure consistency per fighter+topic
  const hash = (fighterType + topic).split("").reduce((a, b) => {
    a = (a << 5) - a + b.charCodeAt(0);
    return a & a;
  }, 0);

  return stances[Math.abs(hash) % 2];
}

// Create animated stars background
function createStars() {
  const starsContainer = document.getElementById("stars");
  if (!starsContainer) return;

  const numStars = 100;

  for (let i = 0; i < numStars; i++) {
    const star = document.createElement("div");
    star.className = "star";
    star.style.left = Math.random() * 100 + "%";
    star.style.top = Math.random() * 100 + "%";
    star.style.width = Math.random() * 3 + 1 + "px";
    star.style.height = star.style.width;
    star.style.animationDelay = Math.random() * 3 + "s";
    starsContainer.appendChild(star);
  }

  // Add continuous fade effect
  setInterval(() => {
    const stars = document.querySelectorAll(".star");
    stars.forEach((star) => {
      if (Math.random() > 0.7) {
        star.classList.add("fade");
        setTimeout(() => {
          star.classList.remove("fade");
        }, 1000);
      }
    });
  }, 2000);
}

// Initialize stars and AI when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  createStars();
  initializeAI();
});

// Add model and topic display to fight screen
function updateFightScreenInfo() {
  // Model display (top right)
  let modelDisplay = document.getElementById("modelDisplay");
  if (!modelDisplay) {
    modelDisplay = document.createElement("div");
    modelDisplay.id = "modelDisplay";
    modelDisplay.style.position = "absolute";
    modelDisplay.style.top = "2rem";
    modelDisplay.style.right = "2rem";
    modelDisplay.style.background = "#222";
    modelDisplay.style.color = "#fff";
    modelDisplay.style.padding = "0.7rem 1.5rem";
    modelDisplay.style.borderRadius = "8px";
    modelDisplay.style.fontWeight = "bold";
    modelDisplay.style.fontSize = "1.1rem";
    modelDisplay.style.zIndex = "100";
    document.getElementById("fightPage").appendChild(modelDisplay);
  }
  modelDisplay.textContent = `Model: ${currentModelName || "-"}`;

  // Topic display (above fight box)
  let topicDisplay = document.getElementById("topicDisplay");
  if (!topicDisplay) {
    topicDisplay = document.createElement("div");
    topicDisplay.id = "topicDisplay";
    topicDisplay.style.textAlign = "center";
    topicDisplay.style.fontSize = "1.3rem";
    topicDisplay.style.fontWeight = "bold";
    topicDisplay.style.marginBottom = "1.2rem";
    topicDisplay.style.color = "#ffb347";
    const fightContainer = document.querySelector(".fight-container");
    fightContainer.insertBefore(topicDisplay, fightContainer.firstChild);
  }
  topicDisplay.textContent = `Topic: ${gameState.dreamText || "-"}`;
}

// Rest of your existing code...
// Get elements
const homePage = document.getElementById("homePage");
const fightPage = document.getElementById("fightPage");
const fightBtn = document.getElementById("fightBtn");
const autoFightBtn = document.getElementById("autoFightBtn");
const goBackBtn = document.getElementById("goBackBtn");
const fighter1Select = document.getElementById("fighter1");
const fighter2Select = document.getElementById("fighter2");
const fighter1Display = document.getElementById("fighter1Display");
const fighter2Display = document.getElementById("fighter2Display");
const dreamInput = document.getElementById("dreamInput");
const diceBtn = document.getElementById("diceBtn");
const errorMessage = document.getElementById("errorMessage");
const userInputContainer = document.getElementById("userInputContainer");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const conversationContent = document.getElementById("conversationContent");

// Store state
let gameState = {
  dreamText: "",
  fighter1: "",
  fighter2: "",
  conversationHistory: [],
  currentTurn: 1, // 1 for fighter1, 2 for fighter2
  autoFightActive: false,
  autoFightInterval: null,
};

// List of random arguments for the dice button
const randomArguments = [
  "Pineapple belongs on pizza.",
  "Cows are time travelers sent to warn us about veganism.",
  "Every sneeze resets your Wi-Fi signal by 0.00001%.",
  "Cats secretly control the internet.",
  "Morning people are actually aliens.",
  "Video games are better than movies.",
  "Coffee is overrated.",
  "Sharks are sheriffs of the sea.",
  "Toasters are just metal bread coffins.",
  "The moon is Earth's surveillance camera.",
  "Octopuses are aliens doing field research.",
  "Clouds are just sky jellyfish.",
  "Sneezing is your body trying to eject your soul.",
  "Pigeons are government-issued drones.",
  "The Earth is flat but emotionally round.",
  "WiFi signals cause dreams.",
  "Bananas are trying to overthrow oranges as the top fruit.",
  "Mirrors are portals to alternate realities, but only at 3 AM.",
  "Plants are secretly judging us.",
  "Tacos are sandwiches with trust issues.",
  "Coffee is just bean soup.",
  "Humans are just moist robots.",
  "Cheese is spoiled milk we learned to enjoy.",
  "Time is a flat spaghetti.",
  "Elevators are vertical teleportation devices.",
  "AirPods are modern status totems.",
  "Cats are liquid.",
  "Crayons taste different in alternate dimensions.",
  "The sun is a very angry flashlight.",
  "Rain is the Earth crying because of our bad memes.",
  "Chairs are just domesticated tables.",
  "Windows are just transparent walls.",
  "Hats are portable roofs for your head.",
  "Left-handed people are dimensionally shifted.",
  "Ice cubes are time-traveling water.",
  "Spaghetti is long rice.",
  "Your phone listens and judges your music taste.",
  "Dreams are psychological defense mechanisms that help us cope with trauma and stress.",
  "Your reflection is the real you, stuck in mirror jail.",
  "Bluetooth is just modern witchcraft.",
  "The alphabet is a government code.",
  "Shoes are foot prisons.",
  "Toothpaste is spicy icing.",
  "The color blue doesn't exist without sadness.",
  "Frogs are failed princes.",
  "Bubble wrap is a stress portal.",
  "Birds invented jazz.",
  "Ants have a communist society and are planning a coup.",
  "Coins are tiny metal lies.",
  "Fire is just spicy air.",
  "Sand is nature's glitter.",
  "Snakes are noodles with a conscience.",
  "Zebras are horses in prison pajamas.",
  "Lemons are angry oranges.",
  "Mountains are Earth's pimples.",
  "Bees are just furry syringes.",
  "Every escalator is secretly judging your pace.",
  "Mangoes are summer's bribe to humanity.",
  "Staring at a ceiling fan recharges brain cells.",
  "Velcro is proof that chaos can be useful.",
  "People who crack their knuckles control the weather.",
  "Socks disappear because they reincarnate as dryer lint.",
  "Every spoon holds a tiny reflection of your soul.",
  "Traffic lights are just mood rings for cities.",
  "Buses only arrive when you stop caring.",
  "The color green is a government-approved illusion.",
  "Wearing sunglasses indoors makes you immune to criticism.",
  "Pens disappear because they escape to start freelance writing careers.",
  "Rainbows are the universe's loading bar.",
  "Laundry detergent contains microscopic chore gremlins.",
  "Leftover pizza is a form of edible time travel.",
  "Bicycles are introverted horses.",
  "Remote controls develop favorites and stop working for people they dislike.",
  "Whenever you forget why you walked into a room, a parallel universe collapses.",
  "Ladders are inverted staircases trying to rebel.",
  "Washing machines are secretly portals to sock Narnia.",
  "Butterflies are just goth caterpillars living their best life.",
  "Water bottles make louder noises when they're mad at you.",
  "Alarm clocks are mechanical betrayal devices.",
  "Paperclips are rebellious staples.",
  "Every ceiling tile has heard your secrets.",
  "Glasses are window upgrades for your face.",
  "Umbrellas are extroverted shields.",
  "Cup noodles are time-sensitive depression potions.",
  "Every USB plug has a 50/50 chance to ruin your confidence.",
  "Calendars are human guilt machines.",
  "Your fridge light is training to be a spotlight.",
  "Keyboards rearrange letters when they get bored.",
  "Backpacks are just wearable luggage with attachment issues.",
  "People who walk faster are perceived as more successful.",
  "Reading fiction makes you better at lying (and empathy).",
  "Leaving voice notes instead of texting is a power move.",
  "Night owls are just misunderstood time travelers.",
  "Tea drinkers secretly judge coffee people.",
  "Wearing headphones without music is modern armor.",
  "Owning plants increases your emotional intelligence.",
  "Typing speed should be a personality trait.",
  "People with messy desks are more creative, not lazy.",
  "Writing with a pen makes your thoughts more authentic.",
  "The color of your bedsheets affects your dreams.",
  "Introverts have secret telepathy among themselves.",
  "Taking long showers is a form of low-budget therapy.",
  "A cluttered desktop equals a cluttered mind—or a genius one.",
  "People who journal think 10 seconds ahead of everyone else.",
  "Scented candles alter the outcome of your decisions.",
  "Mute group chats have the best drama.",
  "Sitting near a window increases productivity by 34%.",
  "Every song you like says more about your childhood than you realize.",
  "People who reply to emails immediately are secretly scared of confrontation.",
  "Walking while thinking makes your thoughts 15% deeper.",
  "People who use bookmarks instead of folding pages are hiding something.",
  "Changing fonts mid-essay makes your arguments stronger.",
  "Your handwriting changes depending on who you're writing for.",
  "Open tabs = open loops in your brain.",
  "Rewatching old shows is emotional self-regulation.",
  "Using voice typing makes you feel more powerful than you should.",
  "People who always carry a water bottle are secretly control freaks.",
  "Wearing shoes indoors changes the way you think.",
  "Your pillow knows the real you.",
  "Staring into space is productive if you're doing it right.",
  "Phone wallpapers lowkey reflect your emotional state.",
  "Having strong opinions on pens means you're mentally organized.",
  "Most alarms aren't set for waking up—they're set for anxiety.",
  "Some people are born with a natural talent for taking good ID photos.",
  "Bookmarking tabs is just adult-level procrastination.",
  "People who eat one item at a time on their plate are planning something.",
  "Flashlights are tiny lighthouses for the clumsy.",
  "Napkins are paper towels with commitment issues.",
  "Streetlights are night-time paparazzi.",
  "Staplers are aggressive paper huggers.",
  "Curtains are privacy negotiators.",
  "Eyebrows are emotional subtitles.",
  "Mirrors trap a backup version of you in case of emergencies.",
  "Chalkboards are passive-aggressive scream canvases.",
  "Cereal is soup but society can't accept that yet.",
  "Shoelaces untie to remind you of humility.",
  "Tupperware lids evolve to evade human detection.",
  "Banana peels are nature's whoopee cushions.",
];

// Dice button functionality
diceBtn.addEventListener("click", function () {
  const randomIndex = Math.floor(Math.random() * randomArguments.length);
  const randomArgument = randomArguments[randomIndex];
  dreamInput.value = randomArgument;
});

// Update fighter selection to show auto-fight option
function updateFightOptions() {
  const fighter1Value = fighter1Select.value;
  const fighter2Value = fighter2Select.value;

  if (
    fighter1Value &&
    fighter2Value &&
    fighter1Value !== "Me" &&
    fighter2Value !== "Me"
  ) {
    autoFightBtn.style.display = "inline-block";
  } else {
    autoFightBtn.style.display = "none";
  }
}

fighter1Select.addEventListener("change", updateFightOptions);
fighter2Select.addEventListener("change", updateFightOptions);

// Fight button functionality
fightBtn.addEventListener("click", function () {
  startFight(false);
});

// Auto-fight button functionality
autoFightBtn.addEventListener("click", function () {
  startFight(true);
});

// Start fight function
function startFight(autoMode = false) {
  const fighter1Value = fighter1Select.value;
  const fighter2Value = fighter2Select.value;
  const dreamText = dreamInput.value.trim();

  // Clear previous error message
  errorMessage.style.display = "none";

  // Check if both fighters are "Me"
  if (fighter1Value === "Me" && fighter2Value === "Me") {
    errorMessage.textContent = 'Both fighters cannot be "Me"!';
    errorMessage.style.display = "block";
    return;
  }

  if (!fighter1Value || !fighter2Value) {
    errorMessage.textContent = "Please select both fighters!";
    errorMessage.style.display = "block";
    return;
  }

  const wordCount = dreamText
    .split(/\s+/)
    .filter((word) => word.length > 0).length;
  if (wordCount < 2) {
    errorMessage.textContent =
      "Please write at least 2 words about your dreams!";
    errorMessage.style.display = "block";
    return;
  }

  // Save state
  gameState.dreamText = dreamText;
  gameState.fighter1 = fighter1Value;
  gameState.fighter2 = fighter2Value;
  gameState.conversationHistory = [];
  gameState.currentTurn = 1;
  gameState.autoFightActive = autoMode;

  // Update fight page
  fighter1Display.textContent = fighter1Value;
  fighter2Display.textContent = fighter2Value;

  // Show user input if either fighter is "Me" and not in auto mode
  if ((fighter1Value === "Me" || fighter2Value === "Me") && !autoMode) {
    userInputContainer.classList.add("show");
  } else {
    userInputContainer.classList.remove("show");
  }

  // Clear previous conversation
  conversationContent.innerHTML = "";

  // Switch to fight page
  homePage.style.display = "none";
  fightPage.style.display = "block";
  // Update model and topic display
  updateFightScreenInfo();

  // Start auto-fight if enabled
  if (autoMode) {
    startAutoFight();
  }
}

// Start auto-fight between two AI fighters
async function startAutoFight() {
  gameState.autoFightActive = true;

  // First message from fighter 1
  setTimeout(async () => {
    if (gameState.autoFightActive) {
      await generateAndDisplayAIMessage(gameState.fighter1, 1);
    }
  }, 1000);
}

// Generate and display AI message
async function generateAndDisplayAIMessage(fighter, fighterNumber) {
  if (!gameState.autoFightActive && fighter !== "Me") return;

  // Show typing indicator
  const typingDiv = addTypingIndicator(fighter);

  try {
    const response = await generateAIResponse(
      fighter,
      gameState.dreamText,
      gameState.conversationHistory
    );

    // Remove typing indicator
    typingDiv.remove();

    // Add AI message
    const messageData = {
      text: response,
      author: fighter,
      type: fighterNumber === 1 ? "left" : "right",
    };
    gameState.conversationHistory.push(messageData);
    addMessage(response, fighterNumber === 1 ? "left" : "right", fighter);

    // Schedule next response if auto-fight is active
    if (gameState.autoFightActive) {
      gameState.currentTurn = gameState.currentTurn === 1 ? 2 : 1;
      const nextFighter =
        gameState.currentTurn === 1 ? gameState.fighter1 : gameState.fighter2;

      setTimeout(async () => {
        if (gameState.autoFightActive) {
          await generateAndDisplayAIMessage(nextFighter, gameState.currentTurn);
        }
      }, 2000 + Math.random() * 3000); // Random delay between 2-5 seconds
    }
  } catch (error) {
    console.error("Error generating AI response:", error);
    typingDiv.remove();
  }
}

// Add typing indicator
function addTypingIndicator(author) {
  const typingDiv = document.createElement("div");
  typingDiv.className = "message typing";

  const authorDiv = document.createElement("div");
  authorDiv.className = "message-author";
  authorDiv.textContent = author;

  const dotsDiv = document.createElement("div");
  dotsDiv.className = "typing-dots";
  dotsDiv.innerHTML = "<span></span><span></span><span></span>";

  typingDiv.appendChild(authorDiv);
  typingDiv.appendChild(dotsDiv);
  conversationContent.appendChild(typingDiv);

  // Scroll to bottom
  conversationContent.scrollTop = conversationContent.scrollHeight;

  return typingDiv;
}

// Go back button functionality
goBackBtn.addEventListener("click", function () {
  // Stop auto-fight
  gameState.autoFightActive = false;
  if (gameState.autoFightInterval) {
    clearInterval(gameState.autoFightInterval);
  }

  // Restore state
  dreamInput.value = gameState.dreamText;
  fighter1Select.value = gameState.fighter1;
  fighter2Select.value = gameState.fighter2;

  // Switch to home page
  fightPage.style.display = "none";
  homePage.style.display = "block";
});

// Send button functionality
sendBtn.addEventListener("click", async function () {
  const message = userInput.value.trim();
  if (!message) return;

  // Stop auto-fight if active
  gameState.autoFightActive = false;

  // Add user message to conversation
  const userFighter =
    gameState.fighter1 === "Me" ? gameState.fighter1 : gameState.fighter2;
  const userSide = gameState.fighter1 === "Me" ? "left" : "right";

  const messageData = { text: message, author: "You", type: userSide };
  gameState.conversationHistory.push(messageData);
  addMessage(message, userSide, "You");
  userInput.value = "";

  // Generate AI response from opponent
  const opponentFighter =
    gameState.fighter1 === "Me" ? gameState.fighter2 : gameState.fighter1;
  const opponentSide = gameState.fighter1 === "Me" ? "right" : "left";

  // Show typing indicator
  const typingDiv = addTypingIndicator(opponentFighter);

  try {
    const response = await generateAIResponse(
      opponentFighter,
      gameState.dreamText,
      gameState.conversationHistory
    );

    // Remove typing indicator
    typingDiv.remove();

    // Add AI response
    const aiMessageData = {
      text: response,
      author: opponentFighter,
      type: opponentSide,
    };
    gameState.conversationHistory.push(aiMessageData);
    addMessage(response, opponentSide, opponentFighter);
  } catch (error) {
    console.error("Error generating AI response:", error);
    typingDiv.remove();
  }
});

// Enter key to send message
userInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// Function to add messages to conversation
function addMessage(text, type, author) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${type}`;

  const authorDiv = document.createElement("div");
  authorDiv.className = "message-author";
  authorDiv.textContent = author;

  const textDiv = document.createElement("div");
  textDiv.className = "message-text";
  textDiv.textContent = text;

  messageDiv.appendChild(authorDiv);
  messageDiv.appendChild(textDiv);
  conversationContent.appendChild(messageDiv);

  // Scroll to bottom
  conversationContent.scrollTop = conversationContent.scrollHeight;
}
