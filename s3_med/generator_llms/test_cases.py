test_cases = [
    {
        "name": "Birth Year - Direct",
        "question": "When was Christopher Columbus born?",
        "context": "Christopher Columbus was born in 1451 in the Republic of Genoa.",
        "answer": "1451"
    },
    {
        "name": "Birth Year - Indirect",
        "question": "When was Christopher Columbus born?",
        "context": "In 1492, Columbus sailed across the Atlantic Ocean. He was 41 years old at the time.",
        "answer": "1451"
    },
    {
        "name": "Birth Year - Misleading",
        "question": "When was Christopher Columbus born?",
        "context": "Christopher Columbus was born in the Republic of Genoa. The Renaissance began in Italy in the 14th century, and Columbus was born during this period of cultural rebirth.",
        "answer": "1451"
    },
    {
        "name": "Invention Year - Direct",
        "question": "When was the telephone invented?",
        "context": "Alexander Graham Bell invented the telephone in 1876.",
        "answer": "1876"
    },
    {
        "name": "Invention Year - Contextual",
        "question": "When was the telephone invented?",
        "context": "The first successful telephone call was made in 1876. Bell's patent was granted on March 7, 1876, just hours before Elisha Gray's application.",
        "answer": "1876"
    },
    {
        "name": "Invention Year - Conflicting",
        "question": "When was the telephone invented?",
        "context": "Antonio Meucci claimed to have invented the telephone in 1854, but couldn't afford the patent. Alexander Graham Bell received the patent in 1876.",
        "answer": "1876"
    },
    {
        "name": "Famous Painting - Direct",
        "question": "Who painted the Mona Lisa?",
        "context": "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.",
        "answer": "Leonardo da Vinci"
    },
    {
        "name": "Famous Painting - Historical",
        "question": "Who painted the Mona Lisa?",
        "context": "During the Renaissance, Leonardo da Vinci created many masterpieces. The Mona Lisa, one of his most famous works, was painted in Florence.",
        "answer": "Leonardo da Vinci"
    },
    {
        "name": "Famous Painting - Misleading",
        "question": "Who painted the Mona Lisa?",
        "context": "The Mona Lisa is displayed in the Louvre Museum in Paris. It was painted during the Renaissance period by a famous Italian artist.",
        "answer": "Leonardo da Vinci"
    },
    {
        "name": "Planet Discovery - Direct",
        "question": "When was Pluto discovered?",
        "context": "Clyde Tombaugh discovered Pluto in 1930.",
        "answer": "1930"
    },
    {
        "name": "Planet Discovery - Scientific",
        "question": "When was Pluto discovered?",
        "context": "After years of searching for Planet X, Clyde Tombaugh discovered Pluto in 1930 using photographic plates at Lowell Observatory.",
        "answer": "1930"
    },
    {
        "name": "Planet Discovery - Confusing",
        "question": "When was Pluto discovered?",
        "context": "Pluto was discovered in 1930. In 2006, it was reclassified as a dwarf planet after the discovery of Eris.",
        "answer": "1930"
    },
    {
        "name": "Scientific Theory - Direct",
        "question": "What is the theory of relativity?",
        "context": "Einstein's theory of relativity describes gravity as the curvature of spacetime and includes the equation E=mc².",
        "answer": "Einstein's theory that describes gravity as the curvature of spacetime and includes the equation E=mc²"
    },
    {
        "name": "Scientific Theory - Technical",
        "question": "What is the theory of relativity?",
        "context": "The theory of relativity, developed by Einstein, revolutionized physics by showing that space and time are interwoven into a single continuum called spacetime.",
        "answer": "Einstein's theory that describes gravity as the curvature of spacetime and includes the equation E=mc²"
    },
    {
        "name": "Scientific Theory - Misleading",
        "question": "What is the theory of relativity?",
        "context": "Einstein's work on relativity built upon Newton's laws of motion. His theory changed our understanding of physics.",
        "answer": "Einstein's theory that describes gravity as the curvature of spacetime and includes the equation E=mc²"
    },
    {
        "name": "Historical Event - Direct",
        "question": "What happened during the first moon landing?",
        "context": "On July 20, 1969, Neil Armstrong became the first human to walk on the moon during the Apollo 11 mission.",
        "answer": "Neil Armstrong became the first human to walk on the moon on July 20, 1969"
    },
    {
        "name": "Historical Event - Detailed",
        "question": "What happened during the first moon landing?",
        "context": "During the Apollo 11 mission, Neil Armstrong and Buzz Aldrin landed on the moon on July 20, 1969. Armstrong was the first to step onto the lunar surface.",
        "answer": "Neil Armstrong became the first human to walk on the moon on July 20, 1969"
    },
    {
        "name": "Historical Event - Confusing",
        "question": "What happened during the first moon landing?",
        "context": "The Apollo program achieved its goal of landing humans on the moon. The first successful landing occurred in 1969 with Armstrong and Aldrin.",
        "answer": "Neil Armstrong became the first human to walk on the moon on July 20, 1969"
    }
]