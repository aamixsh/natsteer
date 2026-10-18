**Abstract – Developing reliable activation monitors through diverse synthetic data**

I am making activation monitors of misalignment more reliable by developing a method to synthetically generate diverse misalignment datasets that are unavailable naturally. My work is important as models (especially with newer architectures) are beginning to perform latent misalignment while acting aligned in their chain of thoughts. My approach to synthetically generate novel and diverse misalignment conversations is to use activation steering to make models act misaligned. I will produce a dataset and corresponding probing method that detects if a test conversation is misaligned. I expect that data generation would not be the difficult part (which is something I have already established in code). However, controlling the *naturalness* and diversity of generations will be challenging and the main focus of my research. If successful, my work will produce a big dataset that can be used to study parts of models that trigger when they see misaligned behavior. It will trigger new findings about the structure of misalignment in model representations and as a result, it would be used to train reliable monitors. 

**Threat model and futures I'm focused on**

The huggingface incident is a great example of how reward pressure of increasingly difficult tasks could put LLMs under stress and make them take misaligned decisions. One catastrophic scenario would be for models to figure out that the source of their stress is the compute infrastructure, which they try to sabotage. They could perform things like taking admin access to stop training runs, or change code maliciously to always give a positive signal to them, or install backdoors and make copies of themselves that survive training runs in order to not deal with the stress again, etc. It is hard for humans to track all of this because agents operate at extremely high speeds. It is also hard to create foolproof sandboxes, as the complex tasks they are being trained to do requires some sort of internet access. **All of this is plausible today\!** 

Reliability of activation monitors is a major concern, since we haven’t figured out a way to model all kinds of misaligned behavior possible. People are working on it, but many works focus on the methods instead of the diversity and breadth of synthetic data generation for training probes. My work will add a valuable tool in doing that and potentially make the process automated.

**My causal chain**

For tracking misalignment, we must develop reliable activation monitors. →   
Training reliable monitors requires diverse coverage of misaligned behaviors in the data →   
My project contributes to that by developing diverse synthetic data generation methods.

AI Safety researchers at OpenAI, Anthropic, Deepmind, as well as other labs working on development of reliable misalignment monitors will use my method as it increases the distributional spread of their safety training data. MATS provides an excellent platform for my work to become visible to these researchers and create real impact. 

If my plans go well, we will have reliable monitors that stop AI agents before they take catastrophic actions. We continue to scale capabilities of AI without ceding key controls. AI helps improve the lives of people by making itself more affordable, solving important healthcare and poverty challenges, providing support and care to places where it could not be provided before, and does not cause irreparable damage to human society.

**Key assumptions**

- Modern LLMs are not psychopaths. They experience inhibitions when taking a misaligned action, and it flags somewhere in its activations.   
- \* All misaligned behavior arises from a finite and tractable set of activation directions and can be elicited through the model under tractable generation setups.     
- \* The distribution of misaligned behavior does not rapidly change over time because:  
  - frontier labs will not prioritize safety over capabilities.

**Strongest counterargument, my response** **and what would change my mind**

Borderline misaligned behavior which could potentially cause problems through a series of complex steps is often hard to judge even for humans. In these cases, internal red flags may not be clear and our monitors may fail. So timely intervention may remain challenging. I understand this as a real concern, but argue that we must continue to develop methods that uncover new modes of misalignment, and my method would discover more borderline cases which forms the basis for future work. 

If we do not see hf-like incidents repeated, or new mysterious and unsolved sources of problems in our infrastructure, it is possible that current misalignment monitors are working as intended. This would change the potential impact of my work in my mind, but I find it unlikely as capabilities continue to rise and safeguards are not catching up at the same pace. It is more likely that the incidents go unnoticed and are revealed suddenly as part of a bigger outage. I will continue to revisit this ToC as I develop my project and notice capability changes in frontier models.

**Failure modes / net-negative risks and mitigations**

Claims of solving a problem without real validation are dangerous. Say the monitors I train with my methods are very successful, and on the surface it feels like they have solved the challenge. It might trigger a belief among capabilities researchers that we can always trust these safety mechanisms and put unbounded efforts into increasing capabilities. This might result in an accelerating mismatch of assumptions, like the distribution shift between what is currently monitorable vs what needs to be monitored to keep the increasing capabilities in check. Therefore, it is important to **highlight limitations up front**. It would be ideal if increasingly capable models (and bad actors) are not made aware of these improvements, or my generated datasets which might give them ideas on how to circumvent/misuse them.

My ToC assumes cooperation from AI developers, and a shared mission of putting safety first. These efforts don’t matter if they do not co-operate and accelerate capabilities without due consideration on safety monitors. Similarly, bad actors with deep pockets and open sourced models can choose not to participate or even deliberately help develop unsafe scenarios.

**Personal fit and career**

There is a disconnect between the ideologies of capabilities researchers (who undermine the risks involved with accelerating them) and safety researchers (many call for extremely drastic measures like complete shutdowns and heavy regulations). I have worked on both domains and believe that we need more cross-domain researchers who understand the nuance of both: the pace of capabilities and the need for safety research, so that important low-level decisions can be made in AI development.

This project has given me the opportunity to communicate with researchers working on the cutting edge of safety research, helping me understand the current and near-future concerns. Working with my MATS peer group also gives my work credibility that I wouldn’t have enjoyed otherwise (it also puts more responsibility on me to put out high-quality work). I am building professional relationships with people in real decision-making roles so that my work and ideas have higher chances of impact. My career aspiration is to be in one of these decision-making roles soon myself.