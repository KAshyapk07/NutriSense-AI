# NutriSense AI — Chrome Web Store Submission Notes

## Short Description (132 chars max)
Indian food nutrition intelligence. Look up any dish, classify food images, and get detailed macro & micronutrient breakdowns.

## Detailed Description

NutriSense AI brings Indian food nutrition intelligence directly to your browser.

**Features:**
- **Instant lookup** — Type any Indian dish (Dal Makhani, Biryani, Paneer Tikka…) and get a full macro & micronutrient breakdown instantly
- **Image classification** — Upload a photo of your food and the AI identifies the dish and its nutritional content
- **Right-click anywhere** — Select text or right-click any food image on any website to look up nutrition without leaving the page
- **Compare dishes** — Ask "Biryani vs Pulao" to get a side-by-side comparison
- **Secure sign-in** — Google OAuth or email/password via Firebase

Powered by a ConvNeXt-Small model trained on 239 Indian food classes and a comprehensive nutrition graph database.

---

## Privacy Policy URL
https://kashyapk07.github.io/NutriSense-AI/Extension/privacy-policy.html

---

## Category
`Lifestyle` (primary) or `Productivity`

## Language
English

---

## Permission Justifications

### `contextMenus`
Adds "Look up nutrition" (for selected text) and "Classify food image" (for images) to the right-click context menu. This is a core workflow — users can analyse food on recipe sites, food delivery apps, and social media without opening the popup.

### `storage`
Stores authentication tokens (access token + refresh token) in `chrome.storage.local` for session persistence. Stores user API preferences in `chrome.storage.sync`. No browsing data is stored.

### `scripting`
Used to inject the nutrition results panel into the active page **only when the user explicitly triggers a context menu action**. The content script is never loaded passively. The panel uses Shadow DOM for complete CSS isolation from the host page.

### `identity`
Used exclusively to launch Google's OAuth 2.0 sign-in flow via `chrome.identity.launchWebAuthFlow`. The extension only requests `openid email profile` scopes — no Drive, Gmail, or other Google service access.

### `host_permissions: <all_urls>`
The context menu features (text lookup and image classification) must work on any website — recipe blogs, food delivery apps (Zomato, Swiggy), social media (Instagram food posts), etc. The content script is injected **only on explicit user action** (right-click → NutriSense menu item); it does not passively load on or read any page.

---

## Single-Purpose Description
NutriSense AI has a single purpose: provide nutritional information for Indian food dishes. All features (text lookup, image classification, context menus, auth) serve this single function.

---

## Data Use Disclosure

| Data type | Collected | Used for | Shared |
|-----------|-----------|----------|--------|
| Personally identifiable information (email, name) | Yes | Authentication | No |
| Authentication info (tokens) | Yes | Session persistence (local only) | No |
| User activity (likes/dislikes) | Yes | Personalisation | No |
| Website content | No | — | — |
| Browsing history | No | — | — |
| Financial info | No | — | — |
| Health info | No | — | — |

All collected data is used only to provide the extension's core functionality. No data is sold or used for advertising.

---

## Screenshots needed (1280×800 or 640×400)
1. Popup — text lookup result for a dish (e.g. Dal Makhani)
2. Popup — image upload / classification result
3. Popup — sign-in screen
4. Context menu — right-click on a food image showing the NutriSense option
5. Context menu result panel injected into a page
