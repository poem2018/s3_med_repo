test_cases = [
    {
        "name": "Birth Year",
        "question": "When was Christopher Columbus born?",
        "context": """
        Christopher Columbus was an Italian explorer and navigator born in 1451 in the Republic of Genoa.
        
        The Renaissance was a period in European history that spanned from the 14th to the 17th century.
        During this time, many great artists like Leonardo da Vinci and Michelangelo created their masterpieces.
        The printing press was invented by Johannes Gutenberg around 1440, revolutionizing the spread of information.
        
        In 1492, Columbus sailed across the Atlantic Ocean, hoping to find a new route to Asia.
        His voyages were sponsored by the Catholic Monarchs of Spain, Ferdinand and Isabella.
        The Spanish Inquisition was established in 1478 to maintain Catholic orthodoxy in Spain.
        
        The Ottoman Empire was expanding during this period, capturing Constantinople in 1453.
        The Black Death had devastated Europe in the 14th century, killing millions of people.
        The Hundred Years' War between England and France ended in 1453.
        """,
        "answer": "1451"
    },
    {
        "name": "Birth Year with Irrelevant Context",
        "question": "When was Christopher Columbus born?",
        "context": """
        The modern smartphone was first introduced in 2007 by Apple.
        The internet became widely available to the public in the 1990s.
        The first electric car was developed in the 19th century.
        
        Quantum computing is a new field of computer science.
        Artificial intelligence has made significant advances in recent years.
        The first successful heart transplant was performed in 1967.
        
        Climate change is a major concern in the 21st century.
        The COVID-19 pandemic began in 2019.
        The first human landing on Mars is planned for the 2030s.
        """,
        "answer": "1451"
    },
    {
        "name": "Invention Year",
        "question": "When was the telephone invented?",
        "context": """
        The 19th century saw many important technological innovations that changed communication forever.
        Samuel Morse invented the telegraph in 1837, allowing messages to be sent over long distances.
        Thomas Edison developed the phonograph in 1877, which could record and play back sound.
        
        Many inventors were working on voice transmission devices in the late 1800s.
        Elisha Gray filed a patent for a telephone-like device on the same day as Alexander Graham Bell.
        Antonio Meucci claimed to have invented the telephone in 1854, but couldn't afford the patent.
        
        The first successful telephone call was made in 1876.
        Bell's patent was granted on March 7, 1876, just hours before Gray's application.
        The first words spoken over the telephone were "Mr. Watson, come here, I want to see you."
        """,
        "answer": "The telephone was invented in 1876"
    },
    {
        "name": "Invention Year with Irrelevant Context",
        "question": "When was the telephone invented?",
        "context": """
        The ancient Egyptians built the pyramids around 2500 BCE.
        The Roman Empire fell in 476 CE.
        The Black Death killed millions in Europe during the 14th century.
        
        The first Olympic Games were held in ancient Greece in 776 BCE.
        The Great Wall of China was built over several centuries.
        The Renaissance began in Italy in the 14th century.
        
        The Industrial Revolution started in Britain in the late 18th century.
        The American Civil War took place from 1861 to 1865.
        The first World War began in 1914.
        """,
        "answer": "The telephone was invented in 1876"
    },
    {
        "name": "Famous Painting",
        "question": "Who painted the Mona Lisa?",
        "context": """
        The Renaissance period produced many famous artists and masterpieces.
        Michelangelo painted the Sistine Chapel ceiling between 1508 and 1512.
        Raphael created the School of Athens fresco in 1511.
        
        Leonardo da Vinci was a polymath who worked in many fields.
        He designed flying machines and studied human anatomy in detail.
        His notebooks contain thousands of pages of drawings and observations.
        
        The Mona Lisa is one of the most famous paintings in the world.
        It was stolen from the Louvre in 1911 and recovered in 1913.
        The painting's enigmatic smile has fascinated viewers for centuries.
        """,
        "answer": "Leonardo da Vinci painted it"
    },
    {
        "name": "Famous Painting with Irrelevant Context",
        "question": "Who painted the Mona Lisa?",
        "context": """
        The first successful airplane flight was made by the Wright brothers in 1903.
        The theory of relativity was developed by Albert Einstein in the early 20th century.
        The first computer was built in the 1940s.
        
        The first human heart transplant was performed in 1967.
        The internet was invented in the 1960s.
        The first mobile phone call was made in 1973.
        
        The first successful cloning of a mammal occurred in 1996.
        The Human Genome Project was completed in 2003.
        The first successful landing on Mars was achieved in 2021.
        """,
        "answer": "Leonardo da Vinci painted it"
    },
    {
        "name": "Planet Discovery",
        "question": "When was Pluto discovered?",
        "context": """
        The outer solar system has been a source of many astronomical discoveries.
        Neptune was discovered in 1846 after mathematical predictions of its existence.
        Uranus was found in 1781 by William Herschel using a telescope.
        
        Many astronomers searched for a ninth planet in the early 20th century.
        Percival Lowell predicted the existence of Planet X based on orbital irregularities.
        Clyde Tombaugh used Lowell's calculations to guide his search for the new planet.
        
        Pluto was reclassified as a dwarf planet in 2006.
        The discovery of Eris in 2005 led to the redefinition of what constitutes a planet.
        Pluto's moon Charon was discovered in 1978.
        """,
        "answer": "Pluto was discovered in 1930"
    },
    {
        "name": "Planet Discovery with Irrelevant Context",
        "question": "When was Pluto discovered?",
        "context": """
        The first successful organ transplant was performed in 1954.
        The first human heart transplant took place in 1967.
        The first successful liver transplant occurred in 1963.
        
        The first successful kidney transplant was performed in 1954.
        The first successful lung transplant was done in 1963.
        The first successful pancreas transplant occurred in 1966.
        
        The first successful intestine transplant was performed in 1987.
        The first successful face transplant took place in 2005.
        The first successful hand transplant occurred in 1998.
        """,
        "answer": "Pluto was discovered in 1930"
    },
    {
        "name": "Scientific Theory",
        "question": "What is the theory of relativity?",
        "context": """
        Albert Einstein revolutionized physics with his theory of relativity in the early 20th century.
        The special theory of relativity, published in 1905, introduced the concept of spacetime.
        The general theory of relativity, published in 1915, described gravity as the curvature of spacetime.
        
        Einstein's work built upon the foundations laid by Isaac Newton's laws of motion.
        The theory of relativity has been confirmed by numerous experiments and observations.
        GPS systems must account for relativistic effects to maintain accuracy.
        
        The theory predicts phenomena like time dilation and length contraction.
        It explains the bending of light around massive objects like stars.
        The famous equation E=mc² comes from the theory of relativity.
        """,
        "answer": "The theory of relativity is Einstein's theory that describes gravity as the curvature of spacetime and includes the famous equation E=mc²"
    },
    {
        "name": "Scientific Theory with Irrelevant Context",
        "question": "What is the theory of relativity?",
        "context": """
        The first successful kidney transplant was performed in 1954.
        The first successful heart transplant took place in 1967.
        The first successful liver transplant occurred in 1963.
        
        Organ transplantation has saved countless lives since its inception.
        Modern immunosuppressive drugs have greatly improved transplant success rates.
        Tissue matching is crucial for successful organ transplantation.
        
        The waiting list for organ transplants continues to grow worldwide.
        Living donor transplants are becoming more common for certain organs.
        Organ donation awareness campaigns have increased donor registration rates.
        """,
        "answer": "The theory of relativity is Einstein's theory that describes gravity as the curvature of spacetime and includes the famous equation E=mc²"
    },
    {
        "name": "Historical Event",
        "question": "What happened during the first moon landing?",
        "context": """
        The Apollo 11 mission was launched on July 16, 1969, with three astronauts aboard.
        Neil Armstrong and Buzz Aldrin landed the lunar module Eagle on the moon's surface.
        Michael Collins remained in orbit aboard the command module Columbia.
        
        Armstrong became the first human to step onto the lunar surface on July 20, 1969.
        His famous words were "That's one small step for man, one giant leap for mankind."
        Aldrin joined Armstrong on the surface about 20 minutes later.
        
        The astronauts collected lunar samples and conducted experiments.
        They spent about 2.5 hours outside the spacecraft.
        The mission returned safely to Earth on July 24, 1969.
        """,
        "answer": "Neil Armstrong and Buzz Aldrin landed on the moon on July 20, 1969, with Armstrong being the first to step on the surface"
    },
    {
        "name": "Historical Event with Irrelevant Context",
        "question": "What happened during the first moon landing?",
        "context": """
        The first successful kidney transplant was performed in 1954.
        The first successful heart transplant took place in 1967.
        The first successful liver transplant occurred in 1963.
        
        Organ transplantation has saved countless lives since its inception.
        Modern immunosuppressive drugs have greatly improved transplant success rates.
        Tissue matching is crucial for successful organ transplantation.
        
        The waiting list for organ transplants continues to grow worldwide.
        Living donor transplants are becoming more common for certain organs.
        Organ donation awareness campaigns have increased donor registration rates.
        """,
        "answer": "Neil Armstrong and Buzz Aldrin landed on the moon on July 20, 1969, with Armstrong being the first to step on the surface"
    }
]