from tanglish.metrics.understanding import (details_heard, invented_numbers, numbers_in,
                                            reply_language)


def test_wrong_name_and_place_are_caught():
    heard = "Hi my name is Rahul. We are planning to visit Alegiri for 4 people next Saturday."
    details = {"name": ["raghu"], "place": ["yelagiri", "elagiri"],
               "guests": ["4"], "date": ["next saturday", "adutha saturday"]}

    assert details_heard(heard, details) == {"name": False, "place": False,
                                             "guests": True, "date": True}


def test_digits_and_words_match_each_other():
    assert details_heard("rooms for four people", {"guests": ["4"]}) == {"guests": True}
    assert details_heard("check in at 8.30 am", {"time": ["8:30"]}) == {"time": True}


def test_detail_must_be_whole_words():
    assert details_heard("visit Elagirikku", {"place": ["elagiri"]}) == {"place": False}


def test_tamil_script_spellings_match_whole_words_only():
    details = {"guests": ["2", "டூ"], "date": ["friday", "ஃப்ரைடே"]}
    assert details_heard("For டூ பீப்பிள். ஃப்ரைடே ரூம்ஸ்", details) == {"guests": True, "date": True}
    # ட is inside கிடைக்குமா; a lost vowel sign would turn டூ into ட and match it
    assert details_heard("ரூம்ஸ் கிடைக்குமா", details) == {"guests": False, "date": False}


def test_numbers_in_reads_prices_times_and_words():
    assert numbers_in("Rs 3,500 per night, check-in 8:30, for four people") == {3500, 8, 30, 4}


def test_invented_numbers_are_those_nobody_gave():
    facts = "Tariff is Rs 3,500 per room per night; early check-in costs Rs 1,000."
    caller = "rooms for 4 people at 8 30"

    reply = "Yes, 4 people at 8:30 is fine; it costs Rs 1,000 plus a Rs 500 deposit."

    assert invented_numbers(reply, [facts, caller]) == [500]


def test_reply_language():
    assert reply_language("Rooms are available on Friday.") == "english"
    assert reply_language("Seri, Friday rooms irukku.") == "tanglish"
    assert reply_language("") == "none"
