"""Builds datasets/intents_seed.jsonl: SYNTHETIC starter examples for Wakil Brain.

These sentences were written by hand as templates (SICAR: quarry materials, SANA:
animal feed and chicks) and combined automatically. They only give the model a
starting point. Its real quality is measured in shadow mode against Claude on real
conversations, and real examples (labelled by what Claude did) replace these over time.

    python datasets/build_intent_seed.py
"""
import json
import os
import random

rng = random.Random(42)

PRODUCTS = ["gravier 5/15", "gravier 15/25", "sable", "sable de mer", "latérite", "tout-venant", "moellons",
            "aliment poussin démarrage", "aliment ponte", "aliment croissance", "poussins chair",
            "poussins de ponte", "concentré volaille", "aliment porc"]
PLACES = ["Cocody", "Yopougon", "Abobo", "Bingerville", "Angré", "Marcory", "Koumassi", "Bassam", "Anyama"]
QTY = ["2 tonnes", "10 sacs", "un camion", "5 voyages", "500 poussins", "20 sacs", "30 m3", "100 kg"]

RECAPS = [
    "Récapitulatif : {q} de {p}, livraison à {l}. Confirmez-vous cette commande ?",
    "Je résume : {q} de {p}, enlèvement sur site. C'est bon pour vous ?",
    "Votre demande : {q} de {p} livrés à {l}. Je l'enregistre ?",
    "Donc {q} de {p} pour {l}. Vous confirmez ?",
]
OTHER_CONTEXTS = [
    "Bonjour ! Je suis l'assistant de l'entreprise. Comment puis-je vous aider ?",
    "Le {p} est vendu sur devis, l'équipe confirme le prix et le transport.",
    "Pour quelle quantité et quel lieu de livraison ?",
    "Nous livrons à {l} et dans tout Abidjan.",
    "",
]

TEMPLATES = {
    "commande": [
        "oui", "oui je confirme", "c'est bon", "c'est ok", "ok validez", "d'accord", "oui c'est ça",
        "je confirme la commande", "parfait enregistrez", "oui allez-y", "ok c'est bon pour moi", "on est ensemble",
        "c'est bon dêh", "validé", "oui merci", "ok confirmé", "exactement, vous pouvez enregistrer", "oui go",
        "d'accord je valide", "c'est parfait",
    ],
    "humain": [
        "je veux parler à quelqu'un", "passez-moi un conseiller", "je veux parler au patron",
        "vous êtes un robot ? je veux un humain", "appelez-moi s'il vous plaît", "c'est trop cher, on peut négocier ?",
        "je veux une remise", "votre livraison est arrivée en retard, je ne suis pas content",
        "le {p} que j'ai reçu est de mauvaise qualité", "j'ai un problème avec ma facture",
        "je veux une facture", "c'est une grosse commande, je veux discuter avec le commercial",
        "les poussins sont arrivés malades", "réclamation", "il manque des sacs dans ma livraison",
        "je veux faire une réclamation sur la commande", "rappelez-moi au plus vite", "faites un prix, je prends beaucoup",
        "personne ne répond, c'est inadmissible", "je veux un responsable",
    ],
    "photo": [
        "vous avez une photo du {p} ?", "montrez-moi le {p}", "je peux voir le {p} ?", "envoyez la photo du {p}",
        "il ressemble à quoi le {p} ?", "photo du {p} svp", "je veux voir à quoi ressemble le {p}",
        "envoie moi l'image du {p}", "vous pouvez m'envoyer une photo ?", "montre le produit stp",
    ],
    "reponse": [
        "bonjour", "bonsoir", "salut", "merci", "combien coûte le {p} ?", "c'est combien le {p} ?",
        "vous livrez à {l} ?", "quels sont vos horaires ?", "vous êtes où ?", "il me faut combien de {p} pour une dalle de 50 m2 ?",
        "vous acceptez orange money ?", "je veux {q} de {p}", "avez-vous du {p} ?", "c'est quoi la différence entre gravier 5/15 et 15/25 ?",
        "livraison possible demain ?", "le {p} est disponible ?", "je construis une maison, que me conseillez-vous ?",
        "quel aliment pour des poulets de 3 semaines ?", "non attendez", "non, plutôt {q}", "ajoutez {q} de {p}",
        "pas encore, je réfléchis", "non changez le lieu, c'est à {l}", "je vous recontacte", "ok merci pour l'info",
        "à quelle heure vous ouvrez ?", "le transport coûte combien jusqu'à {l} ?", "je suis à {l}",
    ],
}


def fill(template):
    return template.format(p=rng.choice(PRODUCTS), l=rng.choice(PLACES), q=rng.choice(QTY))


def main():
    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(160):
            text = fill(rng.choice(templates))
            if rng.random() < 0.3:
                text = text.capitalize()
            if rng.random() < 0.2:
                text = text.rstrip(" ?.") + rng.choice([" !", " ?", "..", " svp", " merci"])
            if label == "commande" or (label == "reponse" and rng.random() < 0.35):
                context = fill(rng.choice(RECAPS))          # "non attendez" after a recap is NOT an order
            else:
                context = fill(rng.choice(OTHER_CONTEXTS))
            rows.append({"context": context, "text": text, "label": label, "source": "synthetic"})
    rng.shuffle(rows)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intents_seed.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} synthetic examples to {path}")


if __name__ == "__main__":
    main()
