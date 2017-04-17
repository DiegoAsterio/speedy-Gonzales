def tidy_this_data(df):
    
    #Changing number attributes into strings as we should not operate with them
    df.loc[df['gender'] == 0, 'gender'] = "Female"
    df.loc[df['gender'] == 1, 'gender'] = "Male"

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['condtn'] == 1, 'condtn'] = "Limited"
    df.loc[df['condtn'] == 2, 'condtn'] = "Extensive"

    #Changing number attributes into boolean as we should not operate with them
    df.loc[df['match'] == 0, 'match'] = False
    df.loc[df['match'] == 1, 'match'] = True

    #Changing number attributes into boolean as we should not operate with them
    df.loc[df['samerace'] == 0, 'samerace'] = False
    df.loc[df['samerace'] == 1, 'samerace'] = True

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['field_cd'] == 1, 'field_cd'] = "Law"
    df.loc[df['field_cd'] == 2, 'field_cd'] = "Math"
    df.loc[df['field_cd'] == 3, 'field_cd'] = "Social Science/Psychologist"
    df.loc[df['field_cd'] == 4, 'field_cd'] = "Medical Science/Pharmaceuticals/Bio Tech"
    df.loc[df['field_cd'] == 5, 'field_cd'] = "Engineering"
    df.loc[df['field_cd'] == 6, 'field_cd'] = "English/Creative Writing/Journalism"
    df.loc[df['field_cd'] == 7, 'field_cd'] = "History/Religion/Philosophy"
    df.loc[df['field_cd'] == 8, 'field_cd'] = "Bussines/Econ/Finance"
    df.loc[df['field_cd'] == 9, 'field_cd'] = "Education/Academia"
    df.loc[df['field_cd'] == 10, 'field_cd'] = "Biological Sciences/Chemistry/Physics"
    df.loc[df['field_cd'] == 11, 'field_cd'] = "Social work"
    df.loc[df['field_cd'] == 12, 'field_cd'] = "Undergrad/undecided"
    df.loc[df['field_cd'] == 13, 'field_cd'] = "Political Science/International Affairs"
    df.loc[df['field_cd'] == 14, 'field_cd'] = "Film"
    df.loc[df['field_cd'] == 15, 'field_cd'] = "Fine Arts/Arts administration"
    df.loc[df['field_cd'] == 16, 'field_cd'] = "Languages"
    df.loc[df['field_cd'] == 17, 'field_cd'] = "Architecture"
    df.loc[df['field_cd'] == 18, 'field_cd'] = "Other"

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['race'] == 1, 'race'] = "Black/African American"
    df.loc[df['race'] == 2, 'race'] = "European/Caucassian-American"
    df.loc[df['race'] == 3, 'race'] = "Latino/Hispanic American"
    df.loc[df['race'] == 4, 'race'] = "Asian/Pacific Islander/Asian-American"
    df.loc[df['race'] == 5, 'race'] = "Native American"
    df.loc[df['race'] == 6, 'race'] = "Other"

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['goal'] == 1, 'goal'] = "Seemed like a fun night"
    df.loc[df['goal'] == 2, 'goal'] = "To meet new people"
    df.loc[df['goal'] == 3, 'goal'] = "To get a date"
    df.loc[df['goal'] == 4, 'goal'] = "Looking for a serious relationship"
    df.loc[df['goal'] == 5, 'goal'] = "To say I did it"
    df.loc[df['goal'] == 6, 'goal'] = "Other"

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['date'] == 1, 'date'] = "Several times a week"
    df.loc[df['date'] == 2, 'date'] = "Twice a week"
    df.loc[df['date'] == 3, 'date'] = "Once a week"
    df.loc[df['date'] == 4, 'date'] = "Twice a month"
    df.loc[df['date'] == 5, 'date'] = "Once a month"
    df.loc[df['date'] == 6, 'date'] = "Several times a year"
    df.loc[df['date'] == 7, 'date'] = "Almost never"

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['go_out'] == 1, 'go_out'] = "Several times a week"
    df.loc[df['go_out'] == 2, 'go_out'] = "Twice a week"
    df.loc[df['go_out'] == 3, 'go_out'] = "Once a week"
    df.loc[df['go_out'] == 4, 'go_out'] = "Twice a month"
    df.loc[df['go_out'] == 5, 'go_out'] = "Once a month"
    df.loc[df['go_out'] == 6, 'go_out'] = "Several times a year"
    df.loc[df['go_out'] == 7, 'go_out'] = "Almost never"

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['career_c'] == 1, 'career_c'] = "Lawyer"
    df.loc[df['career_c'] == 2, 'career_c'] = "Academic/Research"
    df.loc[df['career_c'] == 3, 'career_c'] = "Psychologist"
    df.loc[df['career_c'] == 4, 'career_c'] = "Doctor/Medicine"
    df.loc[df['career_c'] == 5, 'career_c'] = "Engineer"
    df.loc[df['career_c'] == 6, 'career_c'] = "Creative Arts/Entertainment"
    df.loc[df['career_c'] == 7, 'career_c'] = "Banking/Consulting"
    df.loc[df['career_c'] == 8, 'career_c'] = "Real State"
    df.loc[df['career_c'] == 9, 'career_c'] = "International Humanitarian affaires"
    df.loc[df['career_c'] == 10, 'career_c'] = "Undecided"
    df.loc[df['career_c'] == 11, 'career_c'] = "Social work"
    df.loc[df['career_c'] == 12, 'career_c'] = "Speech pathology"
    df.loc[df['career_c'] == 13, 'career_c'] = "Politics"
    df.loc[df['career_c'] == 14, 'career_c'] = "Pro Sports"
    df.loc[df['career_c'] == 15, 'career_c'] = "Other"
    df.loc[df['career_c'] == 16, 'career_c'] = "Journalism"
    df.loc[df['career_c'] == 17, 'career_c'] = "Architecture"

    #Changing number attributes into boolean as we should not operate with them
    df.loc[df['dec'] == 0, 'dec'] = False
    df.loc[df['dec'] == 1, 'dec'] = True

    #Changing number attributes into boolean as we should not operate with them
    df.loc[df['met'] == 1, 'met'] = True
    df.loc[df['met'] == 2, 'met'] = False
    df.loc[df['met'] == 0, 'met'] = False #Presumably they wanted to say False

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['length'] == 1, 'length'] = "Too little"
    df.loc[df['length'] == 2, 'length'] = "Too much"
    df.loc[df['length'] == 3, 'length'] = "Just right"

    #Changing number attributes into strings as we should not operate with them
    df.loc[df['numdat_2'] == 1, 'numdat_2'] = "Too little"
    df.loc[df['numdat_2'] == 2, 'numdat_2'] = "Too much"
    df.loc[df['numdat_2'] == 3, 'numdat_2'] = "Just right"

    #Changing number attributes into boolean as we should not operate with them
    df.loc[df['date_3'] == 0, 'date_3'] = True
    df.loc[df['date_3'] == 1, 'date_3'] = False

    return df

def values_in_a_column(column):
    notRepeated = []
    for value in column:
        if not value in notRepeated:
            notRepeated.append(value)
    return notRepeated

def normalize_data(df):
    N = len(df)
    for i in range(N):
        df.iloc[i,69:75] = df.iloc[i,69:75] * 0.01
    
