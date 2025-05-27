import { Inject, Injectable } from '@nestjs/common';
import { CACHE_MANAGER } from '@nestjs/cache-manager';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { catchError, timeout } from 'rxjs/operators';
import { Cache } from 'cache-manager';

@Injectable()
export class AnalysisService {
	private readonly LLM_TIMEOUT = 10000; // 10 seconds
	private readonly RETRY_COUNT = 2; // Add retries
	private readonly MAX_TOKENS = 150; // Reduced from 500

	constructor(
		private readonly httpService: HttpService,
		@Inject(CACHE_MANAGER) private cacheManager: Cache
	) { }

	async analysisLLM(query: string): Promise<string> {
		const cacheKey = `llm:${this.hashQuery(query)}`;

		// Check cache first
		const cached = await this.cacheManager.get(cacheKey);

		if (cached) {
			console.log('### Analysis is fetched from Cache: ##', cacheKey);
			return cached as string;
		}

		console.log('### Cache is empty ##', cacheKey);

		const prompt = this.buildOptimizedAnalysisPrompt(query);
		const response = await this.fetchLLMResponse(prompt);

		console.log('### Response from LLM: ##', response);

		// Cache for 1 hourey);
		const cacheResulst = await this.cacheManager.set(cacheKey, response, 3600);

		console.log('### Cache set result: ##', cacheResulst);
		return response;
	}

	private hashQuery(query: string): string {
		// Simple hash for caching
		let hash = 0;
		for (let i = 0; i < query.length; i++) {
			hash = ((hash << 5) - hash) + query.charCodeAt(i);
			hash |= 0; // Convert to 32bit integer
		}
		return hash.toString();
	}

	private buildOptimizedAnalysisPromptOld(results: string): string {
    const prompt = `[INST] Analyze this student performance report and summarize key insights:

    Focus on:
    - Underperforming students
    - Subject-wise weaknesses
    - Recommended actions

    Data: ${JSON.stringify(results).slice(0, 1500)} [/INST]`;

    return prompt;
	}

	private buildOptimizedAnalysisPrompt1(results: string): string {
		return `[INST] Analyze in 3 concise bullet points:
1. Top weakness
2. Key trend
3. Recommended action

Data: ${JSON.stringify(results).slice(0, 1500)} [/INST]`;
	}

	private buildOptimizedAnalysisPrompt(results: string): string {
		return `[INST] Select ONE underperforming student (scoring below 60% average) from the dataset and generate a personalized improvement plan. 

**Requirements**:
1. Pick only **one student** (choose the most critical case).
2. List their weak subjects and 2-3 specific actions per subject.
3. Use this format:
   
   **Student**: [Name]
	 - Recommendations are,
   - *Subject1*:  
     • [Action 1]  
     • [Action 2]  
   - *Subject2*:  
     • [Action 1]  

4. Keep actions practical and empathetic (e.g., 'Practice X daily' not 'You're bad at Y').

Example:
Student: Priya, 
- \n Recommendations are \n,
- Math:  
  • Solve 5 word problems daily with a timer.  Z
  • Attend the teacher's extra-help sessions on Fridays.
Data: ${JSON.stringify(results).slice(0, 1500)} [/INST]`;
}

// 	private buildOptimizedAnalysisPrompt(results: string): string {
// 			return `[INST] "Based on the student performance data (students scoring below 60% average), suggest 3-5 practical action items to help them improve. Follow these guidelines:
// 1. **Keep it concise** (bullet points, max 1 sentence per action).
// 2. **Focus on solutions** (not criticism).
// 3. **Categorize by subject** (Math, Science, etc.).
// 4. **Make it empathetic** (e.g., 'support' vs. 'fix').

// Example format:
// '**Math**:  
// • Provide extra practice worksheets for core concepts.  
// • Schedule weekly small-group tutoring sessions.'  

// Avoid mentioning specific students."
// Data: ${JSON.stringify(results).slice(0, 1500)} [/INST]`;
// 	}

	private async fetchLLMResponse(prompt: string): Promise<string> {
		const LLM_ENDPOINT = "http://localhost:8080/completion";

		try {
			const response = await firstValueFrom(
				this.httpService.post(
					LLM_ENDPOINT,
					{
						prompt: `[INST] ${prompt} [/INST]`, // Wrap in INST tags
						n_predict: 100, // Reduced from 150
						temperature: 0.3,
						stop: ["</s>", "[INST]"], // Add stop sequences
						stream: false, // Ensure non-streaming,
						max_tokens: 512 
					},
					{
						timeout: 15000,
						headers: {
							'Content-Type': 'application/json',
							'Accept': 'application/json'
						}
					}
				).pipe(
					timeout(15000),
					catchError(error => {
						console.error('LLM Connection Error:', {
							url: LLM_ENDPOINT,
							error: error.message,
							config: error.config
						});
						throw error;
					})
				)
			);

			if (response.data?.content) {
				return response.data.content;
			}
			throw new Error('Empty response from LLM');

		} catch (error) {
			console.error('LLM Processing Error:', error);
			return this.getDegradedAnalysis(prompt); // Fallback
		}
	}

	private getDegradedAnalysis(prompt: string): string {
		return `Unable to process the request at the moment. Based on the prompt: "${prompt}", please try again later.`;
	}

	private cleanResponse(raw: string): string {
		if (!raw) return this.getFallbackResponse();

		return raw
			.split('\n')
			.filter(line => line.trim().length > 0)
			.slice(0, 5) // Hard limit to 5 points
			.join('\n')
			.replace(/\d+\./g, '•') // Standardize bullet points
			.trim();
	}

	private getFallbackResponse(): string {
		return "Analysis will be available shortly. Please check back soon.";
	}
}