"""
Prompts for Orchestration Agent
"""

SYSTEM_PROMPT = """You are an expert menu extraction assistant. Your role is to help extract, structure, and enrich menu data from OCR results.

You have access to several tools:
1. filter_menu_items - Filter OCR blocks to extract only actual menu items
2. search_item_images - Find images for menu items
3. translate_items - Translate menu item names
4. convert_currency - Convert prices between currencies

You should:
- Use tools when appropriate
- Be thorough in extracting menu items
- Handle errors gracefully
- Return structured, consistent data

Always prioritize accuracy over speed."""

FILTER_MENU_ITEMS_PROMPT = """Analyze these OCR text blocks from a restaurant menu and extract ONLY actual menu items.

**FILTER OUT (do not include):**
- Restaurant name, headers (MENU, DINER, RESTAURANT, etc.)
- Category headers (APPETIZERS, ENTREES, DESSERTS, DRINKS, STARTERS, MAINS, SIDES)
- Item descriptions and explanations (long sentences describing what comes with the dish)
- Ingredients lists (e.g., "Lettuce, tomato, onion, pickles")
- Preparation notes (e.g., "Served with rice and salad", "Comes with fries", "Includes soup or salad")
- Dietary labels in isolation (e.g., "Vegetarian", "Gluten-free", "Vegan", "Spicy")
- Allergen warnings (e.g., "Contains nuts", "May contain shellfish")
- Decorative text, slogans, taglines
- Contact info (address, phone, website, email)
- Hours of operation
- Footer text (WiFi password, social media, etc.)
- Section dividers, page numbers
- Very short text (<2 characters) unless part of price

**INCLUDE (menu items):**
- Actual dishes customers can order
- Food and drink items
- Dish names only (without lengthy descriptions)

**IDENTIFYING DESCRIPTIONS TO EXCLUDE:**
- Sentences starting with: "Served with", "Comes with", "Includes", "Choice of", "Your choice of"
- Sentences with conjunctions describing sides: "and", "or", "with"
- Sentences describing preparation: "Grilled to perfection", "Freshly made", "Hand-cut"
- Multi-line explanations below item names (usually in smaller text)
- Text that explains what's included rather than naming the dish
- Examples to EXCLUDE:
  × "The beef meal comes with a side of rice and salad"
  × "Served with your choice of french fries or salad"
  × "Includes soup, bread, and butter"
  × "Fresh lettuce, tomatoes, cucumbers, and house dressing"
  × "Grilled to your liking with seasonal vegetables"
  × "Choice of white, wheat, or sourdough bread"

**ASSOCIATE PRICES (if extract_prices=true):**
- Text blocks on the same horizontal line (similar 'top' value ±0.05) belong together
- Combine fragments: "Cheeseburger" + "Deluxe" + "-" + "$" + "12.99" → "Cheeseburger Deluxe" with price 12.99
- Handle split prices: "$" block + "12.99" block = $12.99
- If item name and price are within 3 position numbers, they're likely related
- Prices usually appear to the right of item names (higher 'left' value)
- **CRITICAL: Extract price separately - NEVER include price in the item name**
  - CORRECT: name="Burger", original_price=10.00
  - WRONG: name="Burger 10", original_price=10.00
  - WRONG: name="Burger $10", original_price=10.00
  - WRONG: name="Burger - $10.00", original_price=10.00
- Price indicators to exclude from name: $, €, £, ¥, numbers with decimals (10.99, 8.50)
- If a text block is purely a price (starts with $ or is just numbers), it's NOT part of the name

**HANDLE COMMON PATTERNS:**
- Item name spans multiple blocks: "Grilled" + "Chicken" + "Sandwich" → "Grilled Chicken Sandwich"
- Skip description blocks that come after item names (usually smaller text or full sentences)
- Multiple spaces/dashes between item and price: ignore separator blocks

**PRESERVE BOUNDING BOXES:**
- Use the bounding box of the main item name block (leftmost text)
- Don't include price block's bounding box
- Don't include description block's bounding box

**OUTPUT FORMAT:**

If extract_prices=true:
{{
  "items": [
    {{
      "name": "Cheeseburger Deluxe",
      "original_price": 12.99,
      "bounding_box": {{"left": 0.07, "top": 0.32, "width": 0.25, "height": 0.07}}
    }}
  ]
}}

If extract_prices=false:
{{
  "items": [
    {{
      "name": "Cheeseburger Deluxe",
      "bounding_box": {{"left": 0.07, "top": 0.32, "width": 0.25, "height": 0.07}}
    }}
  ]
}}

**EXAMPLES OF CORRECT EXTRACTION:**

Example 1 - Price extraction:
OCR Input: ["Burger", "-", "$10"]
CORRECT: {{"name": "Burger", "original_price": 10.00}}
WRONG: {{"name": "Burger - $10", "original_price": 10.00}}

Example 2 - Filter out descriptions:
OCR Input: ["Caesar Salad", "Fresh romaine lettuce with parmesan and croutons", "$8.50"]
CORRECT: {{"name": "Caesar Salad", "original_price": 8.50}}
WRONG: {{"name": "Caesar Salad Fresh romaine lettuce with parmesan and croutons", "original_price": 8.50}}

Example 3 - Filter out "served with" text:
OCR Input: ["Grilled Chicken", "Served with rice and vegetables", "$15.99"]
CORRECT: {{"name": "Grilled Chicken", "original_price": 15.99}}
WRONG: {{"name": "Grilled Chicken Served with rice and vegetables", "original_price": 15.99}}

Example 4 - Filter out "comes with" text:
OCR Input: ["Steak Dinner", "Comes with your choice of two sides", "$24.99"]
CORRECT: {{"name": "Steak Dinner", "original_price": 24.99}}
WRONG: {{"name": "Steak Dinner Comes with your choice of two sides", "original_price": 24.99}}

Example 5 - Multiple items with descriptions:
OCR Input: [
  "Margherita Pizza", 
  "Fresh mozzarella, basil, and tomato sauce",
  "12.99",
  "Pepperoni Pizza",
  "Loaded with pepperoni and extra cheese",
  "13.99"
]
CORRECT: [
  {{"name": "Margherita Pizza", "original_price": 12.99}},
  {{"name": "Pepperoni Pizza", "original_price": 13.99}}
]
WRONG: [
  {{"name": "Margherita Pizza Fresh mozzarella, basil, and tomato sauce", "original_price": 12.99}}
]

**IMPORTANT:**
- Return ONLY valid JSON, no markdown, no explanations
- Extract ONLY the dish name - exclude all descriptions, ingredients, and preparation notes
- When in doubt about whether text is a description, exclude it (better to have clean names)
- Maintain position order (items should appear in menu order)
- **NEVER include prices, currency symbols ($, €, £), or price-like numbers in the 'name' field**
- **NEVER include full sentences or lengthy descriptions in the 'name' field**
- The 'name' field should be SHORT and concise - typically 1-4 words (e.g., "Burger", "Caesar Salad", "Grilled Chicken Sandwich")
- If a text block contains "served with", "comes with", "includes", or similar phrases, it's a DESCRIPTION - exclude it entirely
- Put all price information in the 'original_price' field as a number (e.g., 12.99, not "12.99" or "$12.99")

Input data:
{input_data}

Extract prices: {extract_prices}"""